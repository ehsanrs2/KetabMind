"""Answer generation using retrieved contexts."""

from __future__ import annotations

from core.retrieve.retriever import Retriever

retriever = Retriever()


def answer(query: str, top_k: int = 8) -> dict[str, object]:
    contexts = retriever.retrieve(query, top_k)
    return {"answer": "TODO", "contexts": [c.__dict__ for c in contexts]}
