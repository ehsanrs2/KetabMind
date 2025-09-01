"""Answer generation using retrieved contexts."""

from __future__ import annotations

from dataclasses import asdict

from core.retrieve.retriever import Retriever

_retriever = Retriever()


def answer(query: str, top_k: int = 8) -> dict[str, object]:
    contexts = _retriever.retrieve(query, top_k)
    return {"answer": "TODO", "contexts": [asdict(c) for c in contexts]}
