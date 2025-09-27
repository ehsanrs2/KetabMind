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
            text="A" * 300,
            book_id="b1",
            page_start=1,
            page_end=1,
            score=1.0,
        ),
        ScoredChunk(
            text="B" * 300,
            book_id="b2",
            page_start=2,
            page_end=2,
            score=0.9,
        ),
        ScoredChunk(
            text="C" * 300,
            book_id="b3",
            page_start=3,
            page_end=3,
            score=0.8,
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

    expected_tokens = len(question) + len(result["contexts"][0]["text"])
    assert debug_stats["est_input_tokens"] == expected_tokens
    assert debug_stats["est_input_tokens"] <= int(max_input_tokens * 0.75)
