"""Context budget helpers for answer selection."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class _CandidateView:
    score: float
    tokens: set[str]
    payload: dict


def _normalize_score(candidate: dict) -> float:
    for key in ("hybrid", "score"):
        value = candidate.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {token.lower() for token in _WORD_RE.findall(text)}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return intersection / union


def _prepare_candidates(candidates: Iterable[dict]) -> list[_CandidateView]:
    prepared: list[_CandidateView] = []
    for candidate in candidates:
        snippet = candidate.get("snippet") or candidate.get("text") or ""
        prepared.append(
            _CandidateView(
                score=_normalize_score(candidate),
                tokens=_tokenize(snippet),
                payload=candidate,
            )
        )
    return prepared


def select_contexts(
    candidates: list[dict],
    max_chunks: int,
    diversity_alpha: float,
) -> list[dict]:
    """Select contexts using a Max Marginal Relevance like approach."""

    if max_chunks <= 0 or not candidates:
        return []

    alpha = max(0.0, min(diversity_alpha, 1.0))
    prepared = _prepare_candidates(candidates)
    prepared.sort(key=lambda c: c.score, reverse=True)

    selected: list[_CandidateView] = []
    remaining = list(prepared)

    while remaining and len(selected) < max_chunks:
        seen_books = {entry.payload.get("book_id") for entry in selected}
        if not selected:
            selected.append(remaining.pop(0))
            continue

        best_index = 0
        best_value = -math.inf
        for index, candidate in enumerate(remaining):
            similarity_penalty = 0.0
            if candidate.tokens:
                similarity_penalty = max(
                    _jaccard_similarity(candidate.tokens, chosen.tokens) for chosen in selected
                )
            book_id = candidate.payload.get("book_id")
            if book_id is not None and book_id in seen_books:
                similarity_penalty += 0.1
            relevance_term = (1.0 - alpha) * candidate.score
            diversity_term = alpha * similarity_penalty
            mmr_score = relevance_term - diversity_term
            if mmr_score > best_value:
                best_value = mmr_score
                best_index = index
        selected.append(remaining.pop(best_index))

    return [entry.payload for entry in selected]


def _normalize_word(word: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", word, flags=re.UNICODE).lower()


def trim_snippet(text: str, question: str, target_tokens: int) -> str:
    """Trim ``text`` around the query terms from ``question``."""

    if target_tokens <= 0:
        return ""

    words = re.findall(r"\S+", text)
    if len(words) <= target_tokens:
        return text.strip()

    question_terms = {
        _normalize_word(token) for token in re.findall(r"\w+", question, flags=re.UNICODE)
    }
    question_terms.discard("")

    match_indices = [
        index for index, word in enumerate(words) if _normalize_word(word) in question_terms
    ]

    if not match_indices:
        start = 0
        end = min(len(words), target_tokens)
    else:
        best_score = (-math.inf, -math.inf, -math.inf)
        best_window = (0, min(len(words), target_tokens))
        for center in match_indices:
            half_window = max(1, target_tokens // 2)
            start = max(0, center - half_window)
            end = min(len(words), start + target_tokens)
            if end - start < target_tokens and start > 0:
                start = max(0, end - target_tokens)
            window_words = words[start:end]
            coverage = sum(1 for token in window_words if _normalize_word(token) in question_terms)
            window_score = (coverage, -abs(center - (start + end) / 2.0), -center)
            if window_score > best_score:
                best_score = window_score
                best_window = (start, end)
        start, end = best_window

    snippet_words = words[start:end]
    snippet = " ".join(snippet_words)

    if start > 0:
        snippet = "…" + snippet
    if end < len(words):
        snippet = snippet + "…"

    return snippet.strip()
