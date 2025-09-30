from __future__ import annotations

import pytest

from core.answer import answerer
from core.retrieve.retriever import ScoredChunk


@pytest.mark.parametrize("max_input_tokens", [512])
def test_answer_trims_contexts_within_budget(
    monkeypatch: pytest.MonkeyPatch, max_input_tokens: int
) -> None:
    contexts = [
        ScoredChunk(
            id="c1",
            book_id="b1",
            page=1,
            snippet="A" * 300,
            cosine=0.0,
            lexical=0.0,
            reranker=0.0,
            hybrid=1.0,
        ),
        ScoredChunk(
            id="c2",
            book_id="b2",
            page=2,
            snippet="B" * 300,
            cosine=0.0,
            lexical=0.0,
            reranker=0.0,
            hybrid=0.9,
        ),
        ScoredChunk(
            id="c3",
            book_id="b3",
            page=3,
            snippet="C" * 300,
            cosine=0.0,
            lexical=0.0,
            reranker=0.0,
            hybrid=0.8,
        ),
    ]

    def fake_retrieve(query: str, top_k: int) -> list[ScoredChunk]:
        return list(contexts)

    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setenv("LLM_MODEL", "mock")
    monkeypatch.setenv("LLM_MAX_INPUT_TOKENS", str(max_input_tokens))
    monkeypatch.setattr(answerer._retriever, "retrieve", fake_retrieve)

    question = "Explain the history of Python programming language."
    result = answerer.answer(question, top_k=3)

    debug_stats = result["debug"]["stats"]
    used_contexts = debug_stats["used_contexts"]
    assert debug_stats["total_contexts"] == len(contexts)
    assert used_contexts == len(result["contexts"])
    assert used_contexts > 0
    assert used_contexts < len(contexts)

    expected_tokens = len(question) + len(result["contexts"][0]["snippet"])
    assert debug_stats["est_input_tokens"] == expected_tokens
    assert debug_stats["est_input_tokens"] <= int(max_input_tokens * 0.75)
