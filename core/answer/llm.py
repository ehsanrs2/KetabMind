"""LLM backend loader and utilities."""

from __future__ import annotations

import atexit
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import nullcontext
from threading import Thread
from typing import Any

import httpx

from core import config


class LLMError(RuntimeError):
    """Base exception for LLM related failures."""


class LLMTimeoutError(LLMError):
    """Raised when the LLM backend does not respond in time."""


class LLMServiceError(LLMError):
    """Raised when the LLM backend returns an unexpected error."""


def _get_env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is not None:
        return value
    return default


def _get_env_bool(key: str, default: bool) -> bool:
    value = _get_env(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _format_http_error(exc: httpx.HTTPError, fallback: str) -> str:
    """Extract a helpful error message from an HTTPX exception."""

    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        text = getattr(response, "text", "")
        if status is not None:
            message = f"HTTP {status} from Ollama"
            if text:
                return f"{message}: {text}"
            return message
        if text:
            return text
    details = str(exc).strip()
    return details or fallback


if not hasattr(httpx, "TimeoutException"):

    class _CompatTimeout(httpx.HTTPError):
        """Backfill TimeoutException for older httpx versions."""

    httpx.TimeoutException = _CompatTimeout  # type: ignore[attr-defined]


_ORIGINAL_HTTPX_POST = getattr(httpx, "post", None)
_ORIGINAL_HTTPX_STREAM = getattr(httpx, "stream", None)


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


class TransformersLLM(LLM):
    """LLM implementation using Hugging Face transformers."""

    def __init__(self) -> None:
        settings = config.settings
        self._model_name = _get_env("LLM_MODEL", settings.llm_model) or settings.llm_model
        self._max_new_tokens = int(_get_env("LLM_MAX_NEW_TOKENS", str(settings.llm_max_new_tokens)))
        self._temperature = float(_get_env("LLM_TEMPERATURE", str(settings.llm_temperature)))
        self._top_p = float(_get_env("LLM_TOP_P", str(settings.llm_top_p)))
        self._torch = self._import_torch()
        default_device = "cpu"
        if self._torch is not None and getattr(self._torch.cuda, "is_available", lambda: False)():
            default_device = "cuda"
        configured_device = _get_env("LLM_DEVICE", default_device)
        self._device = (configured_device or default_device).strip()
        self._load_in_4bit = _get_env_bool("LLM_LOAD_IN_4BIT", True)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._text_streamer_cls: Any | None = None
        self._pad_token_id: int | None = None

    def _import_torch(self) -> Any | None:
        try:
            import torch
        except ImportError:
            return None
        return torch

    def _ensure_loaded(self) -> None:
        if (
            self._tokenizer is not None
            and self._model is not None
            and self._text_streamer_cls is not None
        ):
            return
        if self._torch is None:
            raise LLMServiceError(
                "Transformers backend requires PyTorch. "
                "Install GPU extras via `poetry install --with gpu`."
            )
        if self._device.lower().startswith("cuda") and not self._torch.cuda.is_available():
            raise LLMServiceError(
                f"CUDA device '{self._device}' requested but no GPU is available. "
                "Set LLM_DEVICE=cpu to continue."
            )
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMServiceError(
                "Transformers backend requires the `transformers` package. "
                "Install GPU extras via `poetry install --with gpu`."
            ) from exc

        quantization_config: Any | None = None
        if self._load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise LLMServiceError(
                    "bitsandbytes is required for 4-bit loading. "
                    "Install GPU extras or set LLM_LOAD_IN_4BIT=false."
                ) from exc
            compute_dtype = getattr(self._torch, "float16", None)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype or "float16",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_fast=True)
        self._text_streamer_cls = TextIteratorStreamer

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": "auto",
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name, **model_kwargs)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise LLMServiceError(
                    "Transformers backend ran out of memory while loading the model. "
                    "Try a smaller model or disable 4-bit loading."
                ) from exc
            raise LLMServiceError(f"Failed to load model '{self._model_name}': {exc}") from exc

        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            self._tokenizer.pad_token_id = eos_token_id
            pad_token_id = eos_token_id
        if pad_token_id is None:
            raise LLMServiceError("Tokenizer must define a pad_token_id or eos_token_id.")
        self._pad_token_id = int(pad_token_id)

    def generate(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        encoded = self._tokenizer(prompt, return_tensors="pt")
        if "input_ids" not in encoded:
            raise LLMServiceError("Tokenizer did not return input_ids for the prompt.")
        prompt_length = int(encoded["input_ids"].shape[-1])
        model_inputs = {k: v for k, v in encoded.items()}
        device = None
        if self._torch is not None:
            try:
                device = self._torch.device(self._device)
            except (TypeError, ValueError):
                device = None
        if device is not None:
            try:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            except RuntimeError as exc:
                raise LLMServiceError(
                    f"Failed to move inputs to device '{self._device}': {exc}"
                ) from exc

        do_sample = self._temperature > 0
        generation_kwargs: dict[str, Any] = {
            **model_inputs,
            "max_new_tokens": self._max_new_tokens,
            "top_p": self._top_p,
            "pad_token_id": self._pad_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = self._temperature

        if stream:
            return self._stream_generate(generation_kwargs)
        return self._generate_once(generation_kwargs, prompt_length)

    def _generate_once(self, generation_kwargs: dict[str, Any], prompt_length: int) -> str:
        context = self._torch.inference_mode() if self._torch is not None else nullcontext()
        with context:
            try:
                outputs = self._model.generate(**generation_kwargs)
            except BaseException as exc:  # noqa: BLE001
                self._handle_generation_exception(exc)
        if self._torch is None:
            raise LLMServiceError("PyTorch is required to decode generation outputs.")
        generated = outputs[0][prompt_length:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    def _stream_generate(self, generation_kwargs: dict[str, Any]) -> Iterator[str]:
        assert self._text_streamer_cls is not None
        streamer = self._text_streamer_cls(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        run_kwargs = {**generation_kwargs, "streamer": streamer}
        errors: list[BaseException] = []

        def _worker() -> None:
            context = self._torch.inference_mode() if self._torch is not None else nullcontext()
            with context:
                try:
                    self._model.generate(**run_kwargs)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

        thread = Thread(target=_worker, daemon=True)
        thread.start()

        def iterator() -> Iterator[str]:
            try:
                for chunk in streamer:
                    if chunk:
                        yield chunk
            finally:
                thread.join()
            if errors:
                self._handle_generation_exception(errors[0])

        return iterator()

    def _handle_generation_exception(self, exc: BaseException) -> None:
        message = str(exc)
        if isinstance(exc, RuntimeError) and "out of memory" in message.lower():
            raise LLMServiceError(
                "Transformers backend ran out of memory. "
                "Reduce prompt size, disable 4-bit loading, or select a smaller model."
            ) from exc
        raise LLMServiceError(f"Transformers backend failed: {message}") from exc


class OllamaLLM(LLM):
    """LLM implementation backed by an Ollama server."""

    _timeout_errors = (
        httpx.TimeoutException,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
    )

    def __init__(self) -> None:
        settings = config.settings
        raw_host = _get_env("OLLAMA_HOST", settings.ollama_host)
        if raw_host is None or not raw_host.strip():
            raise LLMServiceError(
                "OLLAMA_HOST is not configured. "
                "Set OLLAMA_HOST to the Ollama server URL (e.g., http://127.0.0.1:11434)."
            )
        normalized_host = raw_host.strip()
        if not normalized_host.lower().startswith(("http://", "https://")):
            normalized_host = f"http://{normalized_host}"
        self._host = normalized_host.rstrip("/")
        self._model = _get_env("LLM_MODEL", settings.llm_model) or settings.llm_model
        self._temperature = float(_get_env("LLM_TEMPERATURE", str(settings.llm_temperature)))
        self._top_p = float(_get_env("LLM_TOP_P", str(settings.llm_top_p)))
        self._max_new_tokens = int(_get_env("LLM_MAX_NEW_TOKENS", str(settings.llm_max_new_tokens)))
        timeout_value = float(_get_env("LLM_REQUEST_TIMEOUT", "60"))
        self._timeout = httpx.Timeout(timeout_value)
        self._endpoint = f"{self._host}/api/generate"
        self._client = httpx.Client(timeout=self._timeout, trust_env=False)
        atexit.register(self._client.close)
        self._post = self._client.post
        self._stream_call = self._client.stream

        patched_post = getattr(httpx, "post", None)
        if patched_post is not None and patched_post is not _ORIGINAL_HTTPX_POST:

            def _patched_post(url: str, *, json: dict[str, Any]) -> httpx.Response:
                return patched_post(  # type: ignore[arg-type]
                    url,
                    json=json,
                    timeout=self._timeout,
                )

            self._post = _patched_post

        patched_stream = getattr(httpx, "stream", None)
        if patched_stream is not None and patched_stream is not _ORIGINAL_HTTPX_STREAM:

            def _patched_stream(method: str, url: str, *, json: dict[str, Any]):
                return patched_stream(  # type: ignore[arg-type]
                    method,
                    url,
                    json=json,
                    timeout=self._timeout,
                )

            self._stream_call = _patched_stream

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
            response = self._post(self._endpoint, json=payload)
            response.raise_for_status()
        except self._timeout_errors as exc:
            raise LLMTimeoutError("Timed out while waiting for Ollama response") from exc
        except httpx.HTTPError as exc:  # pragma: no cover - defensive
            message = _format_http_error(exc, "Failed to contact Ollama")
            raise LLMServiceError(message) from exc
        data = response.json()
        return str(data.get("response", ""))

    def _stream_generate(self, payload: dict[str, object]) -> Iterator[str]:
        def iterator() -> Iterator[str]:
            try:
                with self._stream_call(
                    "POST",
                    self._endpoint,
                    json=payload,
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
            except self._timeout_errors as exc:
                raise LLMTimeoutError("Timed out while streaming from Ollama") from exc
            except httpx.HTTPError as exc:  # pragma: no cover - defensive
                message = _format_http_error(exc, "Ollama streaming request failed")
                raise LLMServiceError(message) from exc

        return iterator()


def get_llm() -> LLM:
    """Instantiate the configured LLM backend."""

    settings = config.settings
    backend = (_get_env("LLM_BACKEND", settings.llm_backend) or settings.llm_backend).lower()
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
