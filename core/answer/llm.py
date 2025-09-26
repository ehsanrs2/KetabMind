"""LLM backend loader and utilities."""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import httpx


class LLMError(RuntimeError):
    """Base exception for LLM related failures."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM backend does not respond in time."""


class LLMServiceError(LLMError):
    """Raised when the LLM backend returns an unexpected error."""


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer, TextIteratorStreamer
else:  # pragma: no cover - optional dependency typing fallback
    PreTrainedModel = Any  # type: ignore[assignment]
    PreTrainedTokenizer = Any  # type: ignore[assignment]
    TextIteratorStreamer = Any  # type: ignore[assignment]


class LLM(ABC):
    """Minimal language model interface."""

    @abstractmethod
    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        """Return a response for the given prompt."""


class MockLLM(LLM):
    """Mock LLM that synthesizes answers from prompt contexts."""

    _BULLET_PATTERN = re.compile(r"^\d+\. \(([^)]+)\)\s+(.*)$")

    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        response = self._build_response(prompt)
        if not stream:
            return response
        return self._stream_response(response)

    def _build_response(self, prompt: str) -> str:
        base = "Mock response generated for testing."
        context_lines = self._extract_context_lines(prompt)
        echoes: list[str] = []
        for idx, line in enumerate(context_lines[:2], 1):
            match = self._BULLET_PATTERN.match(line)
            if match:
                echoes.append(f"Context {idx}: ({match.group(1)}) {match.group(2)}")
            else:
                echoes.append(f"Context {idx}: {line.strip()}")
        if not echoes:
            return base
        return "\n".join([base, *echoes])

    def _extract_context_lines(self, prompt: str) -> list[str]:
        lines = prompt.splitlines()
        context_lines: list[str] = []
        in_ctx = False
        for line in lines:
            if line.strip().startswith("Contexts"):
                in_ctx = True
                continue
            if in_ctx:
                if line.strip().startswith("Answer:"):
                    break
                if line.strip():
                    context_lines.append(line)
        return context_lines

    def _stream_response(self, response: str) -> Iterator[str]:
        for segment in response.split("\n"):
            yield segment + "\n"


def _get_env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class TransformersLLM(LLM):
    """LLM implementation backed by Hugging Face ``transformers``."""

    def __init__(self) -> None:
        self._model_id = os.getenv("LLM_MODEL", "microsoft/Phi-3-mini-4k-instruct")
        self._max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
        self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self._top_p = float(os.getenv("LLM_TOP_P", "0.95"))
        self._load_in_4bit = _get_env_flag("LLM_LOAD_IN_4BIT", True)
        self._requested_device = os.getenv("LLM_DEVICE")
        self._device: str | None = None
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._streamer_cls: type[TextIteratorStreamer] | None = None
        self._torch: Any = None

    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        self._ensure_model()
        assert self._model is not None  # for type-checkers
        assert self._tokenizer is not None
        torch = self._torch
        if torch is None:
            raise LLMServiceError("PyTorch is required for the transformers backend.")
        inputs, input_length = self._prepare_inputs(prompt)
        do_sample = self._temperature > 0
        generation_kwargs = self._build_generation_kwargs(inputs, do_sample)
        if stream:
            return self._stream_generate(generation_kwargs)
        try:
            outputs = self._model.generate(**generation_kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            raise self._convert_exception(exc, "running generation") from exc
        generated = outputs[:, input_length:]
        text = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        return text.strip()

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMServiceError(
                "Transformers backend requires PyTorch. "
                "Install GPU extras via `poetry install --with gpu`."
            ) from exc
        self._torch = torch
        device = self._resolve_device(torch)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise LLMServiceError("Requested CUDA device is not available.")
        self._device = device
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMServiceError(
                "Transformers backend requires the `transformers` package. "
                "Install via `poetry install --with gpu`."
            ) from exc
        quantization_config = None
        if self._load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as exc:  # pragma: no cover - dependency guard
                raise LLMServiceError(
                    "4-bit loading requires bitsandbytes. "
                    "Install GPU extras via `poetry install --with gpu`."
                ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(self._model_id, use_fast=True)
        except Exception as exc:
            raise self._convert_exception(exc, f"loading tokenizer '{self._model_id}'") from exc
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                device_map="auto",
                torch_dtype="auto",
                quantization_config=quantization_config,
            )
        except Exception as exc:
            raise self._convert_exception(exc, f"loading model '{self._model_id}'") from exc
        if (
            getattr(model.config, "pad_token_id", None) is None
            and tokenizer.pad_token_id is not None
        ):
            model.config.pad_token_id = tokenizer.pad_token_id
        self._model = model
        self._tokenizer = tokenizer
        self._streamer_cls = TextIteratorStreamer

    def _prepare_inputs(self, prompt: str) -> tuple[dict[str, Any], int]:
        if self._tokenizer is None or self._device is None:
            raise LLMServiceError("Transformers backend is not initialized.")
        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = encoded.to(self._device)
        input_length = int(encoded["input_ids"].shape[-1])
        return {key: value for key, value in encoded.items()}, input_length

    def _build_generation_kwargs(self, inputs: dict[str, Any], do_sample: bool) -> dict[str, Any]:
        if self._tokenizer is None:
            raise LLMServiceError("Transformers backend is not initialized.")
        kwargs = dict(inputs)
        kwargs["max_new_tokens"] = self._max_new_tokens
        kwargs["pad_token_id"] = self._tokenizer.pad_token_id
        if self._tokenizer.eos_token_id is not None:
            kwargs["eos_token_id"] = self._tokenizer.eos_token_id
        kwargs["do_sample"] = do_sample
        if do_sample:
            kwargs["temperature"] = self._temperature
            kwargs["top_p"] = self._top_p
        return kwargs

    def _stream_generate(self, generation_kwargs: dict[str, Any]) -> Iterator[str]:
        if self._model is None or self._streamer_cls is None or self._tokenizer is None:
            raise LLMServiceError("Transformers backend is not initialized.")
        streamer = self._streamer_cls(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(generation_kwargs)
        kwargs["streamer"] = streamer
        errors: list[Exception] = []

        def runner() -> None:
            try:
                self._model.generate(**kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)
                with contextlib.suppress(Exception):
                    streamer.text_queue.put(None)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

        def iterator() -> Iterator[str]:
            for chunk in streamer:
                if chunk:
                    yield chunk
            thread.join()
            if errors:
                raise self._convert_exception(errors[0], "running generation") from errors[0]

        return iterator()

    def _resolve_device(self, torch: Any) -> str:
        if self._requested_device:
            return self._requested_device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _convert_exception(self, exc: Exception, context: str) -> LLMServiceError:
        message = f"Transformers backend failed while {context}."
        if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            message = (
                "Transformers backend ran out of memory. Try a smaller model, reduce max tokens, "
                "adjust the 4-bit setting, or use a GPU with more VRAM."
            )
        return LLMServiceError(message)

class OllamaLLM(LLM):
    """LLM implementation backed by an Ollama server."""

    def __init__(self) -> None:
        self._host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self._model = os.getenv("LLM_MODEL", "mistral:7b-instruct-q4_K_M")
        self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self._top_p = float(os.getenv("LLM_TOP_P", "0.95"))
        self._max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
        timeout_value = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
        self._timeout = httpx.Timeout(timeout_value)
        self._endpoint = f"{self._host}/api/generate"

    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        options: dict[str, object] = {
            "temperature": self._temperature,
            "top_p": self._top_p,
            "num_predict": self._max_new_tokens,
        }
        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "options": options,
        }
        if stream:
            payload["stream"] = True
            return self._stream_generate(payload)
        return self._generate_once(payload)

    def _generate_once(self, payload: dict[str, object]) -> str:
        try:
            response = httpx.post(self._endpoint, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("Timed out while waiting for Ollama response") from exc
        except httpx.HTTPStatusError as exc:  # pragma: no cover - defensive
            message = exc.response.text or "Ollama returned an HTTP error"
            raise LLMServiceError(message) from exc
        except httpx.HTTPError as exc:  # pragma: no cover - defensive
            raise LLMServiceError("Failed to contact Ollama") from exc
        data = response.json()
        return str(data.get("response", ""))

    def _stream_generate(self, payload: dict[str, object]) -> Iterator[str]:
        def iterator() -> Iterator[str]:
            try:
                with httpx.stream(
                    "POST",
                    self._endpoint,
                    json=payload,
                    timeout=self._timeout,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        delta = chunk.get("response", "")
                        if delta:
                            yield str(delta)
                        if chunk.get("done"):
                            break
            except httpx.TimeoutException as exc:
                raise LLMTimeoutError("Timed out while streaming from Ollama") from exc
            except httpx.HTTPStatusError as exc:  # pragma: no cover - defensive
                message = exc.response.text or "Ollama returned an HTTP error"
                raise LLMServiceError(message) from exc
            except httpx.HTTPError as exc:  # pragma: no cover - defensive
                raise LLMServiceError("Ollama streaming request failed") from exc

        return iterator()


def get_llm() -> LLM:
    """Instantiate the configured LLM backend."""

    backend = os.getenv("LLM_BACKEND", "mock").lower()
    if backend == "mock":
        return MockLLM()
    if backend == "ollama":
        return OllamaLLM()
    if backend == "transformers":
        return TransformersLLM()
    raise ValueError(f"Unsupported LLM_BACKEND: {backend}")


def truncate_to_token_budget(text: str, tokenizer_name: str, max_input_tokens: int) -> str:
    """Truncate ``text`` so that it does not exceed ``max_input_tokens``."""

    if max_input_tokens <= 0:
        return ""
    tokenizer_hint = tokenizer_name.lower()
    pattern = r"\S+\s*"
    if "character" in tokenizer_hint:
        approx = max_input_tokens
        return text[:approx].rstrip()
    end_index: int | None = None
    for token_count, match in enumerate(re.finditer(pattern, text), start=1):
        if token_count > max_input_tokens:
            end_index = match.start()
            break
    if end_index is None:
        return text
    return text[:end_index].rstrip()


def ensure_stop_sequences(text: str, stops: list[str]) -> str:
    """Ensure ``text`` stops at the first occurrence of any sequence in ``stops``."""

    if not stops:
        return text.rstrip()
    earliest = len(text)
    for stop in stops:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1:
            earliest = min(earliest, idx)
    if earliest < len(text):
        return text[:earliest].rstrip()
    return text.rstrip()
