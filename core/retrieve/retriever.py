from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

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
    version: str = ""
    page_num: int = -1
    file_hash: str = ""
    chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever:
    """k-NN retriever with lexical overlap reranking."""

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k
        self.store: VectorStoreLike | None = None

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
        store.ensure_collection(len(qvec))
        hits = store.client.search(collection_name=store.collection, query_vector=qvec, limit=k)

        q_tokens = self._tokenize(query)
        results: list[ScoredChunk] = []
        for hit in hits:
            raw_payload = getattr(hit, "payload", {}) or {}
            payload = cast_mapping(raw_payload)
            text = str(payload.get("text", ""))
            sim = float(getattr(hit, "score", 0.0) or 0.0)
            d = 1 - sim
            overlap = self._overlap_score(q_tokens, self._tokenize(text))
            final = 0.7 * sim + 0.3 * overlap
            version = payload.get("version")
            page_num = payload.get("page_num")
            file_hash = payload.get("file_hash")
            chunk_id = payload.get("chunk_id")
            meta = payload.get("meta")
            if isinstance(meta, Mapping):
                meta_dict = dict(meta)
            else:
                meta_dict = {}
            results.append(
                ScoredChunk(
                    text=text,
                    book_id=str(payload.get("book_id", "")),
                    page_start=int(payload.get("page_start", -1)),
                    page_end=int(payload.get("page_end", -1)),
                    score=final,
                    distance=d,
                    version=str(version) if version is not None else "",
                    page_num=int(page_num) if page_num is not None else int(payload.get("page_start", -1)),
                    file_hash=str(file_hash) if file_hash is not None else "",
                    chunk_id=str(chunk_id) if chunk_id is not None else "",
                    metadata=meta_dict,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]


class SearchClient(Protocol):
    def search(
        self, collection_name: str, query_vector: Sequence[float], limit: int
    ) -> Sequence[Any]: ...


class VectorStoreLike(Protocol):
    client: SearchClient
    collection: str
    def ensure_collection(self, dim: int) -> None: ...


def cast_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    if isinstance(value, dict):  # pragma: no cover - defensive, dict is Mapping
        return cast(Mapping[str, Any], value)
    return cast(Mapping[str, Any], {})
