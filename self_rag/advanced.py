"""Advanced Self-RAG orchestration helpers."""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from types import MethodType
from typing import Any

from core.config import Settings
from core.self_rag.validator import citation_coverage

_WORD_RE = re.compile(r"[\w\u0600-\u06FF']+", re.UNICODE)
_STOPWORDS = {
    "and",
    "the",
    "of",
    "in",
    "on",
    "to",
    "with",
    "for",
    "from",
    "by",
    "at",
}


@dataclass(slots=True)
class AnswerResult:
    """Structured result describing the outcome of an answer synthesis run."""

    question: str
    answer: str
    contexts: list[dict[str, Any]]
    coverage: float
    confidence: float
    passes: int
    reformulated_query: str | None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _ChunkView:
    text: str


def _normalise_token(token: str) -> str:
    return token.strip("'\"").lower()


def _tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        token = _normalise_token(match.group(0))
        if not token or token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _question_terms(question: str) -> set[str]:
    return set(_tokenize(question))


def _average_hybrid(contexts: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for ctx in contexts:
        value = ctx.get("hybrid")
        if value is None:
            value = ctx.get("score")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0.0 or numeric == 0.0:
            values.append(numeric)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _build_chunk_views(contexts: list[dict[str, Any]]) -> list[_ChunkView]:
    views: list[_ChunkView] = []
    for ctx in contexts:
        text = ctx.get("text") or ctx.get("snippet") or ""
        views.append(_ChunkView(text=str(text)))
    return views


def _resolve_int(*candidates: Any, default: int) -> int:
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        return value
    return int(default)


def _resolve_float(*candidates: Any, default: float) -> float:
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value < 0.0:
            continue
        return value
    return float(default)


def _coverage_threshold(settings: Settings) -> float:
    env_value = os.getenv("SELF_RAG_COVERAGE_THRESHOLD")
    candidate = getattr(settings, "self_rag_coverage_threshold", None)
    return _resolve_float(candidate, env_value, default=0.55)


def _topm_delta(settings: Settings) -> int:
    env_value = os.getenv("SELF_RAG_TOPM_DELTA")
    candidate = getattr(settings, "self_rag_topm_delta", None)
    return _resolve_int(candidate, env_value, default=50)


def _lexical_boost(settings: Settings) -> float:
    env_value = os.getenv("SELF_RAG_LEXICAL_BOOST")
    candidate = getattr(settings, "self_rag_lexical_boost", None)
    return _resolve_float(candidate, env_value, default=0.1)


def reformulate_query(question: str, top_chunks: list[dict[str, Any]]) -> str:
    """Return a reformulated query emphasising salient chunk keywords."""

    if not top_chunks:
        return question

    question_terms = _question_terms(question)
    counter: Counter[str] = Counter()
    representatives: dict[str, str] = {}

    for chunk in top_chunks:
        text = chunk.get("text") or chunk.get("snippet") or ""
        for token in _tokenize(str(text)):
            if len(token) < 3:
                continue
            if token in question_terms:
                continue
            if not token.strip():
                continue
            counter[token] += 1
            representatives.setdefault(token, token)

        metadata = chunk.get("metadata")
        if isinstance(metadata, dict):
            entities = metadata.get("entities")
            if isinstance(entities, (list, tuple)):
                for entity in entities:
                    if not isinstance(entity, str):
                        continue
                    token = _normalise_token(entity)
                    if len(token) < 3 or token in question_terms:
                        continue
                    counter[token] += 2
                    representatives.setdefault(token, entity.strip())

    keywords: list[str] = []
    for token, _ in counter.most_common(4):
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= 3:
            break

    if not keywords:
        return question

    chosen = [representatives.get(token, token) for token in keywords]
    return " AND ".join([question, *chosen])


def second_pass(question: str, settings: Settings) -> AnswerResult:
    """Run a second-pass retrieval/answering workflow when coverage is low."""

    from core.answer import answerer  # local import to avoid heavy dependency at module import

    first_result = answerer.answer(question)
    first_contexts: list[dict[str, Any]] = list(first_result.get("contexts", []))
    first_answer = str(first_result.get("answer", ""))
    first_coverage = citation_coverage(first_answer, _build_chunk_views(first_contexts))
    first_confidence = _average_hybrid(first_contexts)

    debug: dict[str, Any] = {
        "first_pass": {
            "answer": first_answer,
            "coverage": first_coverage,
            "confidence": first_confidence,
            "contexts": first_contexts,
            "raw": first_result,
        }
    }

    coverage_threshold = _coverage_threshold(settings)

    if first_coverage >= coverage_threshold or not first_contexts:
        return AnswerResult(
            question=question,
            answer=first_answer,
            contexts=first_contexts,
            coverage=first_coverage,
            confidence=first_confidence,
            passes=1,
            reformulated_query=None,
            debug=debug,
        )

    reformulated = reformulate_query(question, first_contexts)
    retriever = getattr(answerer, "_retriever", None)
    if retriever is None:
        return AnswerResult(
            question=question,
            answer=first_answer,
            contexts=first_contexts,
            coverage=first_coverage,
            confidence=first_confidence,
            passes=1,
            reformulated_query=None,
            debug=debug,
        )

    topm_delta = _topm_delta(settings)
    lexical_delta = _lexical_boost(settings)

    original_top_m = getattr(retriever, "top_m", None)
    original_weights_mapping = getattr(retriever, "hybrid_weights", None)
    original_weights = (
        dict(original_weights_mapping) if original_weights_mapping is not None else None
    )
    original_retrieve = retriever.retrieve
    original_func = getattr(original_retrieve, "__func__", None)

    def _patched_retrieve(self: Any, query: str, top_n: int | None = None):  # type: ignore[override]
        return original_retrieve(reformulated, top_n)

    try:
        if original_top_m is not None and topm_delta:
            retriever.top_m = max(1, int(original_top_m) + int(topm_delta))
        if original_weights is not None and lexical_delta:
            tweaked = dict(original_weights)
            tweaked["lexical"] = tweaked.get("lexical", 0.0) + float(lexical_delta)
            retriever.hybrid_weights = tweaked
        if original_func is not None:
            retriever.retrieve = MethodType(_patched_retrieve, retriever)
        else:
            retriever.retrieve = MethodType(_patched_retrieve, retriever)

        second_result = answerer.answer(question)
    finally:
        if original_top_m is not None:
            retriever.top_m = original_top_m
        if original_weights is not None:
            retriever.hybrid_weights = original_weights
        if original_func is not None:
            retriever.retrieve = MethodType(original_func, retriever)
        else:
            retriever.retrieve = original_retrieve

    second_contexts: list[dict[str, Any]] = list(second_result.get("contexts", []))
    second_answer = str(second_result.get("answer", ""))
    second_coverage = citation_coverage(second_answer, _build_chunk_views(second_contexts))
    second_confidence = _average_hybrid(second_contexts)

    debug["second_pass"] = {
        "answer": second_answer,
        "coverage": second_coverage,
        "confidence": second_confidence,
        "contexts": second_contexts,
        "query": reformulated,
        "raw": second_result,
        "top_m_delta": topm_delta,
        "lexical_boost": lexical_delta,
    }

    return AnswerResult(
        question=question,
        answer=second_answer,
        contexts=second_contexts,
        coverage=second_coverage,
        confidence=second_confidence,
        passes=2,
        reformulated_query=reformulated,
        debug=debug,
    )


__all__ = ["AnswerResult", "reformulate_query", "second_pass"]
