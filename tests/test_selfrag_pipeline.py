from __future__ import annotations

import pytest

from core.answer import answerer
from core.retrieve.retriever import ScoredChunk


def test_second_pass_changes_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_ctx = ScoredChunk(
        id="empty",
        book_id="b",
        page=1,
        snippet="",
        cosine=0.0,
        lexical=0.0,
        reranker=0.0,
        hybrid=0.0,
    )
    good_ctx = ScoredChunk(
        id="good",
        book_id="b",
        page=2,
        snippet="Persian poetry relies heavily on metaphor and rich imagery.",
        cosine=0.0,
        lexical=0.0,
        reranker=0.0,
        hybrid=1.0,
    )

    queries: list[str] = []

    def fake_retrieve(query: str, top_k: int) -> list[ScoredChunk]:
        queries.append(query)
        if "explain further" in query:
            return [good_ctx]
        return [empty_ctx]

    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setattr(answerer._retriever, "retrieve", fake_retrieve)

    result = answerer.answer("What makes Persian poetry unique?", top_k=1)
    ctx_texts = [c["snippet"] for c in result["contexts"]]

    assert queries == [
        "What makes Persian poetry unique?",
        "What makes Persian poetry unique? explain further",
    ]
    assert ctx_texts == [good_ctx.text]
