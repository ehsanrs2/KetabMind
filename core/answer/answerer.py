"""Answer generation using retrieved contexts."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import asdict

from core.retrieve.retriever import Retriever
from core.self_rag.validator import requires_second_pass

from .llm import get_llm
from .template import build_prompt

_retriever = Retriever()


def answer(query: str, top_k: int = 8) -> dict[str, object]:
    contexts = _retriever.retrieve(query, top_k)
    prompt = build_prompt(query, contexts)
    answer_text = "TODO"
    if os.getenv("LLM_BACKEND") == "mock":
        llm = get_llm()
        answer_text = llm.generate(prompt)
        if requires_second_pass(answer_text, contexts):
            expanded = f"{query} explain further"
            contexts = _retriever.retrieve(expanded, top_k)
            prompt = build_prompt(query, contexts)
            answer_text = llm.generate(prompt)
    return {
        "answer": answer_text,
        "contexts": [asdict(c) for c in contexts],
        "debug": {"prompt": prompt},
    }


def stream_answer(query: str, top_k: int = 8) -> Iterable[dict[str, object]]:
    """Yield incremental answer tokens followed by final result."""
    contexts = _retriever.retrieve(query, top_k)
    prompt = build_prompt(query, contexts)
    llm = get_llm()
    text = llm.generate(prompt)
    for token in text.split():
        yield {"delta": f"{token} "}
    yield {"answer": text, "contexts": [asdict(c) for c in contexts]}
