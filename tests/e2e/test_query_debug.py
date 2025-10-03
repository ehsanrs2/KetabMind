from __future__ import annotations

import sys
import types
from typing import Any

import pytest

pytest.importorskip("fastapi")
from core.answer import answerer  # noqa: E402
from core.retrieve.retriever import ScoredChunk  # noqa: E402
from fastapi import URL, Request  # noqa: E402


class StubRetriever:
    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        return [
            ScoredChunk(
                id="chunk-1",
                book_id="book-1",
                page=5,
                snippet="example snippet",
                cosine=0.9,
                lexical=0.1,
                reranker=0.2,
                hybrid=0.6,
                metadata={"version": "v1"},
            )
        ]


class StubLLM:
    def generate(self, prompt: str, stream: bool = False) -> str:
        return "stub-answer"


@pytest.fixture(autouse=True)
def _stub_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        types.SimpleNamespace(asarray=lambda data, dtype=None: data),
    )
    index_stub = types.ModuleType("core.index")

    class _IndexResult:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    index_stub.IndexResult = _IndexResult
    index_stub.find_indexed_file = lambda *args, **kwargs: None
    index_stub.index_path = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "core.index", index_stub)
    monkeypatch.setattr(answerer, "_retriever", StubRetriever())
    monkeypatch.setattr(answerer, "get_llm", lambda: StubLLM())


def test_query_debug_true() -> None:
    from apps.api.main import QueryRequest, query  # noqa: PLC0415

    request = Request("POST", URL("/query"))
    payload = query(QueryRequest(q="question", top_k=1), request=request, stream=False, debug=True)

    citations = payload.get("citations")
    assert citations == ["[book-1:5]"]

    meta = payload.get("meta")
    assert meta == {"lang": "en", "coverage": 0.0, "confidence": 0.6}

    debug = payload.get("debug")
    assert debug is not None
    chunks = debug.get("chunks")
    assert chunks and len(chunks) == 1
    chunk_debug = chunks[0]
    assert chunk_debug == {
        "id": "chunk-1",
        "book_id": "book-1",
        "page_num": 5,
        "cosine": 0.9,
        "lexical": 0.1,
        "reranker": 0.2,
        "hybrid": 0.6,
    }


def test_query_debug_false() -> None:
    from apps.api.main import QueryRequest, query  # noqa: PLC0415

    request = Request("POST", URL("/query"))
    payload = query(QueryRequest(q="question", top_k=1), request=request, stream=False, debug=False)

    assert payload.get("citations") == ["[book-1:5]"]
    assert payload.get("meta") == {"lang": "en", "coverage": 0.0, "confidence": 0.6}
    assert "debug" not in payload
