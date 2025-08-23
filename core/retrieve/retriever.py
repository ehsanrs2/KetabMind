"""Retriever with simple lexical rerank."""

from __future__ import annotations

from dataclasses import dataclass

from core.embed import get_embedder
from core.vector.qdrant import VectorStore, ChunkPayload


@dataclass
class ScoredChunk:
    text: str
    book_id: str
    page_start: int
    page_end: int
    score: float


class Retriever:
    """Retrieve top chunks for a query."""

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k
        self.embedder = get_embedder()
        self.store = VectorStore()

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or self.top_k
        embedding = self.embedder.embed([query])[0]
        hits: list[ChunkPayload] = self.store.query(embedding, k)
        q_tokens = set(query.lower().split())
        scored: list[ScoredChunk] = []
        for hit in hits:
            text = hit["text"]
            tokens = set(text.lower().split())
            overlap = (
                len(q_tokens & tokens) / len(q_tokens | tokens)
                if q_tokens or tokens
                else 0.0
            )
            score = overlap
            scored.append(
                ScoredChunk(
                    text=text,
                    book_id=hit["book_id"],
                    page_start=hit["page_start"],
                    page_end=hit["page_end"],
                    score=score,
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored
