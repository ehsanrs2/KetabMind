from __future__ import annotations

import re
from dataclasses import dataclass

from core.embed import get_embedder
from core.vector.qdrant import VectorStore


@dataclass
class ScoredChunk:
    text: str
    book_id: str
    page_start: int
    page_end: int
    score: float
    distance: float = 0.0


class Retriever:
    """k-NN retriever with lexical overlap reranking."""

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k
        self.store: VectorStore | None = None

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[A-Za-z0-9]+", text.lower()))

    def _overlap_score(self, query_tokens: set[str], chunk_tokens: set[str]) -> float:
        if not query_tokens or not chunk_tokens:
            return 0.0
        common = len(query_tokens & chunk_tokens)
        total = len(query_tokens | chunk_tokens) or 1
        return common / total

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or self.top_k
        embedder = get_embedder()
        qvec = embedder.embed([query])[0]
        store = self.store or VectorStore()
        hits = store.client.search(collection_name=store.collection, query_vector=qvec, limit=k)

        q_tokens = self._tokenize(query)
        results: list[ScoredChunk] = []
        for hit in hits:
            payload = getattr(hit, "payload", {})
            text = payload.get("text", "")
            sim = float(getattr(hit, "score", 0.0))
            d = 1 - sim
            overlap = self._overlap_score(q_tokens, self._tokenize(text))
            final = 0.7 * sim + 0.3 * overlap
            results.append(
                ScoredChunk(
                    text=text,
                    book_id=payload.get("book_id", ""),
                    page_start=int(payload.get("page_start", -1)),
                    page_end=int(payload.get("page_end", -1)),
                    score=final,
                    distance=d,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
