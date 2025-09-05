from __future__ import annotations

import json
import sys
import types

from fastapi.testclient import TestClient

from core.answer import answerer

sys.modules["core.index"] = types.SimpleNamespace(index_path=lambda *args, **kwargs: None)
from apps.api.main import app  # noqa: E402


class DummyRetriever:
    def retrieve(self, query: str, top_k: int):
        return []


class DummyLLM:
    def generate(self, prompt: str) -> str:
        return "foo bar baz"


def test_query_stream(monkeypatch) -> None:
    monkeypatch.setattr(answerer, "_retriever", DummyRetriever())
    monkeypatch.setattr(answerer, "get_llm", lambda: DummyLLM())
    client = TestClient(app)
    with client.stream("POST", "/query?stream=true", json={"q": "x", "top_k": 3}) as resp:
        frames = [json.loads(line) for line in resp.iter_lines() if line]
    assert frames[-1]["answer"] == "foo bar baz"
    deltas = [f for f in frames[:-1] if "delta" in f]
    assert len(deltas) >= 2
