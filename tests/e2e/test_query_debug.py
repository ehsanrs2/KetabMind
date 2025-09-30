from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from core.answer import answerer
from core.retrieve.retriever import ScoredChunk


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
    monkeypatch.setattr(answerer, "_retriever", StubRetriever())
    monkeypatch.setattr(answerer, "get_llm", lambda: StubLLM())


def test_query_debug_true() -> None:
    from apps.api.main import app  # noqa: PLC0415

    client = TestClient(app)
    response = client.post("/query?debug=true", json={"q": "question", "top_k": 1})
    assert response.status_code == 200
    payload = response.json()

    citations = payload.get("citations")
    assert citations and citations[0]["page_num"] == 5

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
    from apps.api.main import app  # noqa: PLC0415

    client = TestClient(app)
    response = client.post("/query", json={"q": "question", "top_k": 1})
    assert response.status_code == 200
    payload = response.json()

    assert payload.get("citations")
    assert "debug" not in payload
