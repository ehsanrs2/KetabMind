"""Unit tests for the Ollama LLM backend using mocked HTTP responses."""

from __future__ import annotations

import json as json_module
from collections.abc import Iterator
from types import TracebackType
from typing import Any

import httpx
import pytest

from core.answer.llm import LLMServiceError, LLMTimeoutError, OllamaLLM


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _DummyStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __enter__(self) -> _DummyStream:  # pragma: no cover - trivial
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:  # pragma: no cover - trivial
        return None

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None

    def iter_lines(self) -> Iterator[str]:
        yield from self._lines


def _configure_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("LLM_MODEL", "mock-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    monkeypatch.setenv("LLM_TOP_P", "0.8")
    monkeypatch.setenv("LLM_MAX_NEW_TOKENS", "42")


def test_ollama_llm_non_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)
    captured_payload: dict[str, Any] = {}

    def fake_post(url: str, *, json: dict[str, Any], timeout: Any) -> _DummyResponse:
        assert url == "http://localhost:11434/api/generate"
        captured_payload.update(json)
        return _DummyResponse({"response": "hello world"})

    monkeypatch.setattr(httpx, "post", fake_post)

    llm = OllamaLLM()
    result = llm.generate("prompt text", stream=False)

    assert isinstance(result, str)
    assert result == "hello world"
    assert captured_payload["model"] == "mock-model"
    assert captured_payload["options"]["num_predict"] == 42


def test_ollama_llm_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)
    payloads: list[dict[str, Any]] = []

    def fake_stream(method: str, url: str, *, json: dict[str, Any], timeout: Any) -> _DummyStream:
        assert method == "POST"
        assert url == "http://localhost:11434/api/generate"
        payloads.append(json)
        chunks = [
            json_module.dumps({"response": "Hello", "done": False}),
            "",
            json_module.dumps({"response": " world", "done": False}),
            json_module.dumps({"response": "!", "done": True}),
        ]
        return _DummyStream(chunks)

    monkeypatch.setattr(httpx, "stream", fake_stream)

    llm = OllamaLLM()
    stream = llm.generate("prompt text", stream=True)

    assert not isinstance(stream, str)
    deltas = list(stream)

    assert len(deltas) >= 2
    assert "".join(deltas) == "Hello world!"
    assert payloads[0]["stream"] is True


def test_ollama_llm_stream_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)

    def fake_stream(method: str, url: str, *, json: dict[str, Any], timeout: Any) -> None:
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(httpx, "stream", fake_stream)

    llm = OllamaLLM()
    stream = llm.generate("prompt text", stream=True)
    assert not isinstance(stream, str)
    with pytest.raises(LLMTimeoutError):
        next(stream)


def test_ollama_llm_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)

    def fake_post(url: str, *, json: dict[str, Any], timeout: Any) -> None:
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(httpx, "post", fake_post)

    llm = OllamaLLM()
    with pytest.raises(LLMTimeoutError):
        llm.generate("prompt text", stream=False)


def test_ollama_llm_service_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)

    def fake_post(url: str, *, json: dict[str, Any], timeout: Any) -> None:
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx, "post", fake_post)

    llm = OllamaLLM()
    with pytest.raises(LLMServiceError):
        llm.generate("prompt text", stream=False)
