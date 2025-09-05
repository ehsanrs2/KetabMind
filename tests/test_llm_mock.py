from __future__ import annotations

import re

import pytest

from core.answer import answerer
from core.retrieve.retriever import ScoredChunk


def test_llm_mock_uses_context(monkeypatch: pytest.MonkeyPatch) -> None:
    top_ctx = ScoredChunk(
        text="Python is a programming language. It is widely used.",
        book_id="b1",
        page_start=1,
        page_end=1,
        score=1.0,
    )

    def fake_retrieve(query: str, top_k: int) -> list[ScoredChunk]:
        return [top_ctx]

    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setattr(answerer._retriever, "retrieve", fake_retrieve)

    result = answerer.answer("What is Python?", top_k=1)
    answer = str(result["answer"])
    assert answer != "TODO"
    tokens = set(re.findall(r"\w+", top_ctx.text.lower()))
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    assert tokens & answer_tokens
