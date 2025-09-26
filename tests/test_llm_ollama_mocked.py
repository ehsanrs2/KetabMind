"""Unit tests for the Ollama LLM backend using mocked HTTP responses."""

from __future__ import annotations

import json as json_module
from collections.abc import Iterator
from types import TracebackType
from typing import Any

import httpx
import pytest

from core.answer.llm import OllamaLLM


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

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert kwargs.get("trust_env") is False

        def post(self, url: str, json: dict[str, Any]) -> _DummyResponse:
            assert url == "http://localhost:11434/api/generate"
            captured_payload.update(json)
            return _DummyResponse({"response": "hello world"})

        def stream(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - safety
            raise AssertionError("stream should not be called in non-stream mode")

        def close(self) -> None:  # pragma: no cover - no-op
            return None

    monkeypatch.setattr(httpx, "Client", _FakeClient)

    llm = OllamaLLM()
    result = llm.generate("prompt text", stream=False)

    assert isinstance(result, str)
    assert result == "hello world"
    assert captured_payload["model"] == "mock-model"
    assert captured_payload["options"]["num_predict"] == 42


def test_ollama_llm_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_env(monkeypatch)
    payloads: list[dict[str, Any]] = []

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert kwargs.get("trust_env") is False

        def post(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - safety
            raise AssertionError("post should not be called in stream mode")

        def stream(self, method: str, url: str, *, json: dict[str, Any]) -> _DummyStream:
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

        def close(self) -> None:  # pragma: no cover - no-op
            return None

    monkeypatch.setattr(httpx, "Client", _FakeClient)

    llm = OllamaLLM()
    stream = llm.generate("prompt text", stream=True)

    assert not isinstance(stream, str)
    deltas = list(stream)

    assert len(deltas) >= 2
    assert "".join(deltas) == "Hello world!"
    assert payloads[0]["stream"] is True
