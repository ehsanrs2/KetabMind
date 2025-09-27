from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from types import ModuleType
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from core.answer import answerer


def _noop_index_path(*args: object, **kwargs: object) -> None:
    return None


def _ensure_core_index_stub() -> None:
    if "core.index" in sys.modules:
        return
    fake_index_module: ModuleType = types.ModuleType("core.index")
    module_as_any = cast(Any, fake_index_module)
    module_as_any.index_path = _noop_index_path
    sys.modules["core.index"] = fake_index_module


class DummyRetriever:
    def retrieve(self, query: str, top_k: int) -> list[Any]:
        return []


class ExplodingLLM:
    def generate(self, prompt: str, stream: bool = False) -> Iterator[str]:
        if not stream:
            raise RuntimeError("boom")

        def iterator() -> Iterator[str]:
            raise RuntimeError("boom")
            yield ""  # pragma: no cover - unreachable

        return iterator()


def test_streaming_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(answerer, "_retriever", DummyRetriever())
    monkeypatch.setattr(answerer, "get_llm", lambda: ExplodingLLM())
    _ensure_core_index_stub()
    from apps.api.main import app  # noqa: PLC0415

    client = TestClient(app)
    with client.stream("POST", "/query?stream=true", json={"q": "x", "top_k": 2}) as resp:
        frames = [json.loads(line) for line in resp.iter_lines() if line]
    assert frames[0]["error"] == "boom"
    assert len(frames) == 1
