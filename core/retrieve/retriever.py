"""Retriever with hybrid scoring pipeline."""
from __future__ import annotations
import hashlib
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast
import structlog
from caching import LRUCache
from core.config import settings
from core.embed import get_embedder
from core.fts import get_backend as get_fts_backend
from core.vector.qdrant import VectorStore
from lexical import overlap_score
from qdrant_client.http import models as rest

if TYPE_CHECKING:
    from reranker import RerankerAdapter
else:
    RerankerAdapter = Any

@dataclass
class ScoredChunk:
    id: str
    book_id: str
    page: int
    snippet: str
    cosine: float
    lexical: float
    reranker: float
    hybrid: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str: return self.snippet
    @property
    def score(self) -> float: return self.hybrid
    @property
    def page_start(self) -> int: return self.page
    @property
    def page_end(self) -> int: return self.page
    @property
    def distance(self) -> float: return 1.0 - self.hybrid

@dataclass
class _Candidate:
    id: str
    book_id: str
    page: int
    snippet: str
    cosine: float
    lexical: float = 0.0
    reranker: float = 0.0
    hybrid: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_chunk(self) -> ScoredChunk:
        return ScoredChunk(
            id=self.id, book_id=self.book_id, page=self.page, snippet=self.snippet,
            cosine=self.cosine, lexical=self.lexical, reranker=self.reranker,
            hybrid=self.hybrid, metadata=dict(self.metadata),
        )

class SearchClient(Protocol):
    def search(self, collection_name: str, query_vector: Sequence[float], limit: int, query_filter: Any | None = None) -> Sequence[Any]: ...

class VectorStoreLike(Protocol):
    client: SearchClient
    collection: str
    def ensure_collection(self, dim: int) -> None: ...

def _cast_mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, (Mapping, dict)) else {}

def _parse_weights(raw: str) -> dict[str, float]:
    weights: MutableMapping[str, float] = {"cosine": 0.4, "lexical": 0.2, "reranker": 0.4}
    if raw:
        for item in raw.split(","):
            name, _, value = item.partition("=")
            if name.strip():
                try: weights[name.strip().lower()] = float(value)
                except ValueError: continue
    return dict(weights)

log = structlog.get_logger(__name__)

class Retriever:
    def __init__(self, *, top_k: int | None = None, top_n: int | None = None, top_m: int = 200, reranker_topk: int | None = None, hybrid_weights: Mapping[str, float] | None = None, embedder: Any | None = None, store: VectorStoreLike | None = None, reranker: RerankerAdapter | None = None, reranker_enabled: bool | None = None, reranker_cache_size: int | None = None) -> None:
        self.top_n = int(top_n or top_k or 10)
        self.top_m = max(1, top_m)
        self.reranker_topk = reranker_topk or settings.reranker_topk
        self.hybrid_weights = dict(hybrid_weights or _parse_weights(settings.hybrid_weights))
        self.embedder = embedder
        self.store = store
        self._reranker = reranker
        self.reranker_enabled = bool(reranker or settings.reranker_enabled) if reranker_enabled is None else reranker_enabled
        cache_size = max(0, int(reranker_cache_size if reranker_cache_size is not None else settings.reranker_cache_size))
        self._reranker_cache = LRUCache(cache_size) if cache_size > 0 else None

    def _ensure_store(self, dim: int) -> VectorStoreLike:
        if self.store is None: self.store = VectorStore()
        self.store.ensure_collection(dim)
        return self.store

    def _get_embedder(self) -> Any:
        if self.embedder is None: self.embedder = get_embedder()
        return self.embedder

    def _get_reranker(self) -> RerankerAdapter | None:
        if not self.reranker_enabled: return None
        if self._reranker is None:
            from reranker import RerankerAdapter as RerankerAdapterCls
            self._reranker = RerankerAdapterCls(settings.reranker_model_name, device=None, batch_size=max(1, settings.reranker_batch))
        return self._reranker

    def _vector_hits(self, store: VectorStoreLike, query_vector: Sequence[float], limit: int, page_map: Mapping[str, set[int]] | None, book_id_filter: str | None) -> list[Any]:
        # [اصلاحیه] اعمال فیلتر book_id حتی اگر page_map وجود نداشته باشد
        base_filter = None
        if book_id_filter:
            base_filter = rest.Filter(must=[rest.FieldCondition(key="book_id", match=rest.MatchValue(value=book_id_filter))])

        if not page_map:
            return list(store.client.search(
                collection_name=store.collection,
                query_vector=query_vector,
                limit=limit,
                query_filter=base_filter  # اعمال فیلتر در جستجوی عادی
            ))

        hits: list[Any] = []
        multiplier = max(1, settings.fts_vector_multiplier)
        for matched_book_id, pages in page_map.items():
            # اگر فیلتر کلی وجود دارد و با کتاب پیدا شده در FTS مغایرت دارد، نادیده بگیر
            if book_id_filter and matched_book_id != book_id_filter:
                continue
                
            try:
                book_hits = store.client.search(
                    collection_name=store.collection,
                    query_vector=query_vector,
                    limit=max(limit, len(pages) * multiplier),
                    query_filter=rest.Filter(
                        must=[rest.FieldCondition(key="book_id", match=rest.MatchValue(value=matched_book_id))]
                    ),
                )
                hits.extend(book_hits)
            except Exception:
                log.warning("retriever.fts_vector_search_failed", book_id=matched_book_id, exc_info=True)
        return hits

    def _build_candidates(self, query: str, hits: Sequence[Any], page_map: Mapping[str, set[int]] | None) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for hit in hits:
            payload = _cast_mapping(getattr(hit, "payload", {}))
            snippet = str(payload.get("text", ""))
            book_id = str(payload.get("book_id", ""))
            # لاگیک ساده شده برای کاندیداها
            cosine = float(getattr(hit, "score", 0.0) or 0.0)
            chunk_id = payload.get("chunk_id") or getattr(hit, "id", "")
            cand = _Candidate(id=str(chunk_id), book_id=book_id, page=int(payload.get("page_start", -1)), snippet=snippet, cosine=cosine, metadata=dict(_cast_mapping(payload.get("meta"))))
            cand.lexical = overlap_score(query, snippet)
            candidates.append(cand)
        return candidates

    def _hybrid_score(self, candidate: _Candidate) -> float:
        # (همان منطق قبلی)
        components = [("cosine", candidate.cosine), ("lexical", candidate.lexical)]
        if candidate.reranker: components.append(("reranker", candidate.reranker))
        active = {k: v for k, v in self.hybrid_weights.items() if any(c[0] == k for c in components) and v != 0}
        if not active: return candidate.cosine
        total = sum(active.values())
        if total == 0: return candidate.cosine
        score = sum(active.get(name, 0) * value for name, value in components)
        return score / total if total != 1 else score

    def retrieve(self, query: str, top_n: int | None = None, top_k: int | None = None, *, book_id: str | None = None) -> list[ScoredChunk]:
        final_n = int(top_n or top_k or self.top_n)
        embedder = self._get_embedder()
        query_vector = embedder.embed([query])[0] # فراخوانی متد embed
        store = self._ensure_store(len(query_vector))

        initial_limit = max(final_n, self.top_m, self.reranker_topk)
        page_map: dict[str, set[int]] = {}
        
        # FTS Logic
        fts = get_fts_backend()
        if fts.is_available():
            try:
                matches = fts.search(query, book_id=book_id, limit=max(1, settings.fts_page_limit))
                for m in matches: 
                    if m.page_num >= 0: page_map.setdefault(m.book_id, set()).add(int(m.page_num))
            except Exception:
                log.warning("retriever.fts_failed", exc_info=True)

        # [اصلاحیه] پاس دادن book_id به _vector_hits
        hits = self._vector_hits(store, query_vector, initial_limit, page_map if page_map else None, book_id_filter=book_id)
        candidates = self._build_candidates(query, hits, page_map if page_map else None)

        if page_map and not candidates:
            # Fallback to pure vector search with filter
            hits = self._vector_hits(store, query_vector, initial_limit, None, book_id_filter=book_id)
            candidates = self._build_candidates(query, hits, None)

        # Reranking Logic (simplified for brevity, logic remains same)
        reranker = self._get_reranker()
        if reranker and candidates:
            rerank_cands = sorted(candidates, key=lambda c: c.lexical, reverse=True)[: min(self.reranker_topk, len(candidates))]
            pairs = [(query, c.snippet) for c in rerank_cands]
            try:
                scores = reranker.score_pairs(pairs)
                for cand, score in zip(rerank_cands, scores): cand.reranker = float(score)
            except Exception: log.warning("reranker.failed", exc_info=True)

        for c in candidates: c.hybrid = self._hybrid_score(c)
        return [c.to_chunk() for c in sorted(candidates, key=lambda c: c.hybrid, reverse=True)[:final_n]]

__all__ = ["Retriever", "ScoredChunk"]