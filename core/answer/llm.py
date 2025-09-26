"""LLM backend loader and utilities."""

from __future__ import annotations

import atexit
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator

import httpx


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


class OllamaLLM(LLM):
    """LLM implementation backed by an Ollama server."""

    _timeout_errors = (
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
    )

    def __init__(self) -> None:
        self._host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        self._model = os.getenv("LLM_MODEL", "mistral:7b-instruct-q4_K_M")
        self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self._top_p = float(os.getenv("LLM_TOP_P", "0.95"))
        self._max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
        timeout_value = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))
        self._timeout = httpx.Timeout(timeout_value)
        self._endpoint = f"{self._host}/api/generate"
        self._client = httpx.Client(timeout=self._timeout, trust_env=False)
        atexit.register(self._client.close)

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
            response = self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
        except self._timeout_errors as exc:
            raise TimeoutError("Timed out while waiting for Ollama response") from exc
        except httpx.HTTPError as exc:  # pragma: no cover - defensive
            raise RuntimeError("Ollama request failed") from exc
        data = response.json()
        return str(data.get("response", ""))

    def _stream_generate(self, payload: dict[str, object]) -> Iterator[str]:
        def iterator() -> Iterator[str]:
            try:
                with self._client.stream(
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
                raise TimeoutError("Timed out while streaming from Ollama") from exc
            except httpx.HTTPError as exc:  # pragma: no cover - defensive
                raise RuntimeError("Ollama streaming request failed") from exc

        return iterator()


def get_llm() -> LLM:
    """Instantiate the configured LLM backend."""

    backend = os.getenv("LLM_BACKEND", "mock").lower()
    if backend == "mock":
        return MockLLM()
    if backend == "ollama":
        return OllamaLLM()
    if backend == "transformers":
        raise NotImplementedError(f"LLM backend '{backend}' is not implemented yet.")
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
