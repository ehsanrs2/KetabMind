"""Retriever with hybrid scoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, Protocol, Sequence, cast

from core.config import settings
from core.embed import get_embedder
from core.vector.qdrant import VectorStore
from lexical import overlap_score
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
        self, collection_name: str, query_vector: Sequence[float], limit: int
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


class Retriever:
    """Three-stage retriever combining vector, lexical, and reranker signals."""

    def __init__(
        self,
        *,
        top_n: int | None = None,
        top_m: int = 200,
        reranker_topk: int | None = None,
        hybrid_weights: Mapping[str, float] | None = None,
        embedder: Any | None = None,
        store: VectorStoreLike | None = None,
        reranker: RerankerAdapter | None = None,
        reranker_enabled: bool | None = None,
    ) -> None:
        self.top_n = top_n or 10
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

    def retrieve(self, query: str, top_n: int | None = None) -> list[ScoredChunk]:
        final_n = top_n or self.top_n
        embedder = self._get_embedder()
        query_vector = embedder.embed([query])[0]
        store = self._ensure_store(len(query_vector))

        initial_limit = max(final_n, self.top_m, self.reranker_topk)
        hits = store.client.search(
            collection_name=store.collection,
            query_vector=query_vector,
            limit=initial_limit,
        )

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

        reranker = self._get_reranker()
        if reranker and candidates:
            rerank_candidates = sorted(
                candidates, key=lambda c: c.lexical, reverse=True
            )[: min(self.reranker_topk, len(candidates))]
            pairs = [(query, c.snippet) for c in rerank_candidates]
            scores = reranker.score_pairs(pairs)
            for cand, score in zip(rerank_candidates, scores):
                cand.reranker = float(score)

        for candidate in candidates:
            candidate.hybrid = self._hybrid_score(candidate)

        ranked = sorted(candidates, key=lambda c: c.hybrid, reverse=True)
        return [cand.to_chunk() for cand in ranked[:final_n]]


__all__ = ["Retriever", "ScoredChunk"]
