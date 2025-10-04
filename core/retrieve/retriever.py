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

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from reranker import RerankerAdapter
else:  # pragma: no cover - runtime placeholder
    RerankerAdapter = Any  # type: ignore[misc, assignment]


@dataclass
class ScoredChunk:
    """Final retrieval result with per-signal scores."""

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
    def text(self) -> str:
        return self.snippet

    @property
    def score(self) -> float:
        return self.hybrid

    @property
    def page_start(self) -> int:
        return self.page

    @property
    def page_end(self) -> int:
        return self.page

    @property
    def distance(self) -> float:
        return 1.0 - self.hybrid


@dataclass
class _Candidate:
    """Intermediate candidate keeping all score components."""

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
            id=self.id,
            book_id=self.book_id,
            page=self.page,
            snippet=self.snippet,
            cosine=self.cosine,
            lexical=self.lexical,
            reranker=self.reranker,
            hybrid=self.hybrid,
            metadata=dict(self.metadata),
        )


class SearchClient(Protocol):
    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int,
        query_filter: Any | None = None,
    ) -> Sequence[Any]: ...


class VectorStoreLike(Protocol):
    client: SearchClient
    collection: str

    def ensure_collection(self, dim: int) -> None: ...


def _cast_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    if isinstance(value, dict):  # pragma: no cover - defensive
        return cast(Mapping[str, Any], value)
    return cast(Mapping[str, Any], {})


def _parse_weights(raw: str) -> dict[str, float]:
    weights: MutableMapping[str, float] = {"cosine": 0.4, "lexical": 0.2, "reranker": 0.4}
    if not raw:
        return dict(weights)
    for item in raw.split(","):
        name, _, value = item.partition("=")
        name = name.strip().lower()
        if not name:
            continue
        try:
            weights[name] = float(value)
        except ValueError:
            continue
    return dict(weights)


log = structlog.get_logger(__name__)


class Retriever:
    """Three-stage retriever combining vector, lexical, and reranker signals."""

    def __init__(
        self,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        top_m: int = 200,
        reranker_topk: int | None = None,
        hybrid_weights: Mapping[str, float] | None = None,
        embedder: Any | None = None,
        store: VectorStoreLike | None = None,
        reranker: RerankerAdapter | None = None,
        reranker_enabled: bool | None = None,
        reranker_cache_size: int | None = None,
    ) -> None:
        if top_n is not None and top_k is not None or top_n is not None:
            self.top_n = int(top_n)
        elif top_k is not None:
            self.top_n = int(top_k)
        else:
            self.top_n = 10
        self.top_m = max(1, top_m)
        self.reranker_topk = reranker_topk or settings.reranker_topk
        self.hybrid_weights = dict(hybrid_weights or _parse_weights(settings.hybrid_weights))
        self.embedder = embedder
        self.store = store
        self._reranker = reranker
        if reranker_enabled is None:
            self.reranker_enabled = bool(reranker or settings.reranker_enabled)
        else:
            self.reranker_enabled = reranker_enabled
        if reranker_cache_size is None:
            reranker_cache_size = max(0, int(settings.reranker_cache_size))
        if reranker_cache_size > 0:
            self._reranker_cache: LRUCache[tuple[str, str], float] | None = LRUCache(
                reranker_cache_size
            )
        else:
            self._reranker_cache = None

    def _ensure_store(self, dim: int) -> VectorStoreLike:
        if self.store is None:
            self.store = VectorStore()
        self.store.ensure_collection(dim)
        return self.store

    def _get_embedder(self) -> Any:
        if self.embedder is None:
            self.embedder = get_embedder()
        return self.embedder

    def _get_reranker(self) -> RerankerAdapter | None:
        if not self.reranker_enabled:
            return None
        if self._reranker is None:
            from reranker import RerankerAdapter as RerankerAdapterCls  # lazy import

            self._reranker = RerankerAdapterCls(
                settings.reranker_model_name,
                device=None,
                batch_size=max(1, settings.reranker_batch),
            )
        return self._reranker

    def _vector_hits(
        self,
        store: VectorStoreLike,
        query_vector: Sequence[float],
        limit: int,
        page_map: Mapping[str, set[int]] | None,
    ) -> list[Any]:
        if not page_map:
            return list(
                store.client.search(
                    collection_name=store.collection,
                    query_vector=query_vector,
                    limit=limit,
                )
            )

        hits: list[Any] = []
        multiplier = max(1, settings.fts_vector_multiplier)
        for matched_book_id, pages in page_map.items():
            try:
                book_hits = store.client.search(
                    collection_name=store.collection,
                    query_vector=query_vector,
                    limit=max(limit, len(pages) * multiplier),
                    query_filter=rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key="book_id",
                                match=rest.MatchValue(value=matched_book_id),
                            )
                        ]
                    ),
                )
            except Exception:
                log.warning(
                    "retriever.fts_vector_search_failed",
                    book_id=matched_book_id,
                    exc_info=True,
                )
                continue
            hits.extend(book_hits)
        return hits

    def _build_candidates(
        self,
        query: str,
        hits: Sequence[Any],
        page_map: Mapping[str, set[int]] | None,
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for hit in hits:
            payload = _cast_mapping(getattr(hit, "payload", {}))
            snippet = str(payload.get("text", ""))
            book_id = str(payload.get("book_id", ""))
            page_raw = payload.get("page_num")
            if page_raw is None:
                page_raw = payload.get("page_start", -1)
            try:
                page = int(page_raw)
            except (TypeError, ValueError):
                page = -1

            if page_map is not None:
                allowed = page_map.get(book_id)
                if not allowed or page not in allowed:
                    continue

            cosine = float(getattr(hit, "score", 0.0) or 0.0)
            chunk_id = payload.get("chunk_id") or getattr(hit, "id", "")
            candidate = _Candidate(
                id=str(chunk_id),
                book_id=book_id,
                page=page,
                snippet=snippet,
                cosine=cosine,
                metadata=dict(_cast_mapping(payload.get("meta"))),
            )
            candidate.lexical = overlap_score(query, snippet)
            candidates.append(candidate)
        return candidates

    def _hybrid_score(self, candidate: _Candidate) -> float:
        components: list[tuple[str, float]] = [
            ("cosine", candidate.cosine),
            ("lexical", candidate.lexical),
        ]
        if candidate.reranker:
            components.append(("reranker", candidate.reranker))

        active_weights = {
            name: weight
            for name, weight in self.hybrid_weights.items()
            if any(comp[0] == name for comp in components) and weight != 0.0
        }
        if not active_weights:
            # fallback to cosine only
            return candidate.cosine

        total_weight = sum(active_weights.values())
        if total_weight == 0:
            return candidate.cosine

        score = 0.0
        for name, value in components:
            weight = active_weights.get(name)
            if weight is None:
                continue
            score += weight * value
        return score / total_weight if total_weight != 1 else score

    def retrieve(
        self,
        query: str,
        top_n: int | None = None,
        top_k: int | None = None,
        *,
        book_id: str | None = None,
    ) -> list[ScoredChunk]:
        if top_n is not None and top_k is not None or top_n is not None:
            final_n = int(top_n)
        elif top_k is not None:
            final_n = int(top_k)
        else:
            final_n = self.top_n
        embedder = self._get_embedder()
        query_vector = embedder.embed([query])[0]
        store = self._ensure_store(len(query_vector))

        initial_limit = max(final_n, self.top_m, self.reranker_topk)
        page_map: dict[str, set[int]] = {}
        fts_backend = get_fts_backend()
        if fts_backend.is_available():
            try:
                matches = fts_backend.search(
                    query,
                    book_id=book_id,
                    limit=max(1, settings.fts_page_limit),
                )
            except Exception:
                log.warning("retriever.fts_failed", exc_info=True)
            else:
                for match in matches:
                    if match.page_num < 0:
                        continue
                    page_map.setdefault(match.book_id, set()).add(int(match.page_num))

        hits = self._vector_hits(
            store,
            query_vector,
            initial_limit,
            page_map if page_map else None,
        )
        candidates = self._build_candidates(
            query,
            hits,
            page_map if page_map else None,
        )

        if page_map and not candidates:
            hits = self._vector_hits(store, query_vector, initial_limit, None)
            candidates = self._build_candidates(query, hits, None)

        reranker = self._get_reranker()
        if reranker and candidates:
            rerank_candidates = sorted(candidates, key=lambda c: c.lexical, reverse=True)[
                : min(self.reranker_topk, len(candidates))
            ]
            cache = self._reranker_cache
            query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
            pending_pairs: list[tuple[_Candidate, tuple[str, str], tuple[str, str]]] = []
            for cand in rerank_candidates:
                cache_key = (query_hash, cand.id)
                if cache:
                    cached = cache.get(cache_key, None)
                    if cached is not None:
                        cand.reranker = float(cached)
                        continue
                pending_pairs.append((cand, cache_key, (query, cand.snippet)))

            if pending_pairs:
                pairs = [pair for _, _, pair in pending_pairs]
                try:
                    scores = reranker.score_pairs(pairs)
                except TimeoutError:
                    log.warning(
                        "reranker.timeout",
                        pair_count=len(pairs),
                        timeout=getattr(reranker, "timeout_s", None),
                    )
                else:
                    if len(scores) != len(pending_pairs):
                        log.warning(
                            "reranker.score_mismatch",
                            expected=len(pending_pairs),
                            received=len(scores),
                        )
                    for (cand, cache_key, _), score in zip(pending_pairs, scores, strict=False):
                        cand.reranker = float(score)
                        if cache:
                            cache.put(cache_key, cand.reranker)

        for candidate in candidates:
            candidate.hybrid = self._hybrid_score(candidate)

        ranked = sorted(candidates, key=lambda c: c.hybrid, reverse=True)
        return [cand.to_chunk() for cand in ranked[:final_n]]


__all__ = ["Retriever", "ScoredChunk"]
