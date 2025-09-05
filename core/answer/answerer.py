"""Answer generation using retrieved contexts."""

from __future__ import annotations

from dataclasses import asdict

from core.retrieve.retriever import Retriever

from .template import build_prompt

_retriever = Retriever()


def answer(query: str, top_k: int = 8) -> dict[str, object]:
    contexts = _retriever.retrieve(query, top_k)
    prompt = build_prompt(query, contexts)
    return {
        "answer": "TODO",
        "contexts": [asdict(c) for c in contexts],
        "debug": {"prompt": prompt},
    }
