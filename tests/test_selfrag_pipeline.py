from __future__ import annotations

import pytest

from core.answer import answerer
from core.retrieve.retriever import ScoredChunk


def test_second_pass_changes_contexts(monkeypatch: pytest.MonkeyPatch) -> None:
    empty_ctx = ScoredChunk(text="", book_id="b", page_start=1, page_end=1, score=0.0)
    good_ctx = ScoredChunk(
        text="Persian poetry relies heavily on metaphor and rich imagery.",
        book_id="b",
        page_start=2,
        page_end=2,
        score=1.0,
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
    ctx_texts = [c["text"] for c in result["contexts"]]

    assert queries == [
        "What makes Persian poetry unique?",
        "What makes Persian poetry unique? explain further",
    ]
    assert ctx_texts == [good_ctx.text]
