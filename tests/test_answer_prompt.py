from __future__ import annotations

from core.answer.template import build_prompt
from core.retrieve.retriever import ScoredChunk


def _ctx() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            id="c1",
            book_id="b1",
            page=1,
            snippet="A",
            cosine=0.0,
            lexical=0.0,
            reranker=0.0,
            hybrid=0.0,
        ),
        ScoredChunk(
            id="c2",
            book_id="b2",
            page=3,
            snippet="B",
            cosine=0.0,
            lexical=0.0,
            reranker=0.0,
            hybrid=0.0,
        ),
    ]


def test_prompt_has_question_count_and_citation_hint() -> None:
    q = "What is X?"
    ctx = _ctx()
    prompt = build_prompt(q, ctx, "System test")
    assert q in prompt
    assert str(len(ctx)) in prompt
    assert "[book_id:page_start-page_end]" in prompt
    assert "(b1:1)" in prompt
