from __future__ import annotations

import json
import sys
import types
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


class DummyLLM:
    def generate(self, prompt: str) -> str:
        return "foo bar baz"


def test_query_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(answerer, "_retriever", DummyRetriever())
    monkeypatch.setattr(answerer, "get_llm", lambda: DummyLLM())
    _ensure_core_index_stub()
    from apps.api.main import app  # noqa: PLC0415

    client = TestClient(app)
    with client.stream("POST", "/query?stream=true", json={"q": "x", "top_k": 3}) as resp:
        frames = [json.loads(line) for line in resp.iter_lines() if line]
    assert frames[-1]["answer"] == "foo bar baz"
    deltas = [f for f in frames[:-1] if "delta" in f]
    assert len(deltas) >= 2
