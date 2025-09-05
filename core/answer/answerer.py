"""Answer generation using retrieved contexts."""

from __future__ import annotations

import os
from dataclasses import asdict

from core.retrieve.retriever import Retriever

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
    return {
        "answer": answer_text,
        "contexts": [asdict(c) for c in contexts],
        "debug": {"prompt": prompt},
    }
