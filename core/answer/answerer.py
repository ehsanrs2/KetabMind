"""Answer generation using retrieved contexts."""

from __future__ import annotations

from dataclasses import asdict

from core.retrieve.retriever import Retriever
from core.self_rag.validator import validate

_retriever = Retriever()


def answer(query: str, top_k: int = 8) -> dict[str, object]:
    contexts = _retriever.retrieve(query, top_k)
    ans = "TODO"

    # Validate grounding against retrieved contexts
    if not validate(ans, contexts):
        # Simple self-critique: reformulate query and retry retrieval
        alt_query = f"{query} elaborate"
        contexts = _retriever.retrieve(alt_query, top_k)

    return {"answer": ans, "contexts": [asdict(c) for c in contexts]}
