"""Self-RAG orchestration utilities."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, MutableMapping, Protocol, Sequence, cast

from core.self_rag.validator import citation_coverage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.retrieve.retriever import Retriever as _RetrieverType, ScoredChunk as _ScoredChunk
else:  # pragma: no cover - fallback for optional dependency loading
    _RetrieverType = Any
    _ScoredChunk = Any

ScoredChunk = _ScoredChunk


class RetrieverLike(Protocol):
    """Minimal protocol required from retrievers used by the runner."""

    top_m: int
    hybrid_weights: MutableMapping[str, float]

    def retrieve(self, query: str, top_n: int | None = None) -> list[ScoredChunk]:
        ...


class Synthesizer(Protocol):
    """Callable that turns a query and contexts into an answer."""

    def __call__(self, query: str, contexts: Sequence[ScoredChunk]) -> str:
        ...


@dataclass(slots=True)
class SelfRAGResult:
    """Final outcome of a Self-RAG run."""

    query: str
    answer: str
    contexts: list[ScoredChunk]
    coverage: float
    avg_hybrid: float
    passes: int
    reformulated_query: str | None = None
    debug: dict[str, object] = field(default_factory=dict)


_WORD_RE = re.compile(r"[\w\-']+", re.UNICODE)


def _average_hybrid_score(contexts: Sequence[ScoredChunk]) -> float:
    if not contexts:
        return 0.0
    return sum(chunk.hybrid for chunk in contexts) / len(contexts)


def _extract_keywords(contexts: Sequence[ScoredChunk], *, limit: int = 3) -> list[str]:
    """Return the most frequent keywords from the highest-scoring chunks."""

    if not contexts:
        return []
    sorted_chunks = sorted(contexts, key=lambda c: c.hybrid, reverse=True)
    counter: Counter[str] = Counter()
    for chunk in sorted_chunks[: max(1, min(len(sorted_chunks), limit))]:
        words = _WORD_RE.findall(chunk.text.lower())
        for word in words:
            if len(word) < 4:
                continue
            counter[word] += 1
    keywords: list[str] = []
    for word, _ in counter.most_common(limit):
        if word not in keywords:
            keywords.append(word)
    return keywords


def _reformulate_query(original: str, contexts: Sequence[ScoredChunk], *, keyword_limit: int) -> str:
    keywords = _extract_keywords(contexts, limit=keyword_limit)
    if keywords:
        extra = " ".join(keywords)
        if extra.strip():
            return f"{original} {extra}"
    return f"{original} more details"


def _tweak_hybrid_weights(weights: MutableMapping[str, float], delta: float) -> MutableMapping[str, float]:
    updated: MutableMapping[str, float] = {k: float(v) for k, v in weights.items()}
    if delta == 0:
        return updated
    updated["lexical"] = updated.get("lexical", 0.0) + delta
    cosine_penalty = delta / 2
    if cosine_penalty:
        updated["cosine"] = max(0.0, updated.get("cosine", 0.0) - cosine_penalty)
    return updated


def run_self_rag(
    query: str,
    *,
    retriever: RetrieverLike | None = None,
    synthesizer: Synthesizer,
    top_k: int = 8,
    coverage_threshold: float = 0.5,
    hybrid_threshold: float = 0.4,
    keyword_limit: int = 3,
    second_pass_top_m: int | None = None,
    hybrid_weight_delta: float = 0.1,
) -> SelfRAGResult:
    """Execute a Self-RAG workflow with optional second retrieval pass."""

    if retriever is None:
        from core.retrieve.retriever import Retriever as RetrieverCls

        retriever = cast("RetrieverLike", RetrieverCls())

    contexts = retriever.retrieve(query, top_k)
    answer = synthesizer(query, contexts)
    coverage = citation_coverage(answer, contexts)
    avg_hybrid = _average_hybrid_score(contexts)

    debug: dict[str, object] = {
        "first_pass": {
            "coverage": coverage,
            "avg_hybrid": avg_hybrid,
            "answer": answer,
        }
    }

    if coverage >= coverage_threshold and avg_hybrid >= hybrid_threshold:
        return SelfRAGResult(
            query=query,
            answer=answer,
            contexts=list(contexts),
            coverage=coverage,
            avg_hybrid=avg_hybrid,
            passes=1,
            reformulated_query=None,
            debug=debug,
        )

    reformulated_query = _reformulate_query(query, contexts, keyword_limit=keyword_limit)
    original_top_m = getattr(retriever, "top_m", None)
    original_weights = getattr(retriever, "hybrid_weights", None)

    try:
        if original_top_m is not None:
            target_top_m = second_pass_top_m or int(math.ceil(original_top_m * 1.5))
            retriever.top_m = max(original_top_m, target_top_m)
        if original_weights is not None:
            tweaked = _tweak_hybrid_weights(original_weights, hybrid_weight_delta)
            retriever.hybrid_weights = tweaked

        second_contexts = retriever.retrieve(reformulated_query, top_k)
    finally:
        if original_top_m is not None:
            retriever.top_m = original_top_m
        if original_weights is not None:
            retriever.hybrid_weights = original_weights

    second_answer = synthesizer(query, second_contexts)
    second_coverage = citation_coverage(second_answer, second_contexts)
    second_avg_hybrid = _average_hybrid_score(second_contexts)

    debug["second_pass"] = {
        "coverage": second_coverage,
        "avg_hybrid": second_avg_hybrid,
        "answer": second_answer,
        "query": reformulated_query,
    }

    return SelfRAGResult(
        query=query,
        answer=second_answer,
        contexts=list(second_contexts),
        coverage=second_coverage,
        avg_hybrid=second_avg_hybrid,
        passes=2,
        reformulated_query=reformulated_query,
        debug=debug,
    )


__all__ = ["run_self_rag", "SelfRAGResult"]
