from __future__ import annotations

from core.answer.template import build_prompt
from core.retrieve.retriever import ScoredChunk


def _ctx() -> list[ScoredChunk]:
    return [
        ScoredChunk(text="A", book_id="b1", page_start=1, page_end=2, score=0.0),
        ScoredChunk(text="B", book_id="b2", page_start=3, page_end=4, score=0.0),
    ]


def test_prompt_has_question_count_and_citation_hint() -> None:
    q = "What is X?"
    ctx = _ctx()
    prompt = build_prompt(q, ctx)
    assert q in prompt
    assert str(len(ctx)) in prompt
    assert "cite" in prompt.lower()
    assert "(b1 1-2)" in prompt
