"""Lexical overlap utilities for hybrid retrieval."""

from __future__ import annotations

import re

from core.config import settings
from nlp.fa_normalize import normalize_fa

__all__ = ["overlap_score"]

_FA_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")
_TOKEN_PATTERN = re.compile(r"[\w\u200c]+", re.UNICODE)

_FA_STOPWORDS = {
    "و",
    "به",
    "در",
    "از",
    "که",
    "این",
    "آن",
    "برای",
    "را",
    "با",
    "تا",
    "اما",
    "هم",
    "هر",
    "نیز",
    "است",
    "شد",
    "کرد",
    "می",
}

_EN_STOPWORDS = {
    "the",
    "and",
    "or",
    "to",
    "of",
    "a",
    "an",
    "in",
    "on",
    "for",
    "is",
    "it",
    "with",
    "as",
    "by",
    "be",
    "are",
    "was",
    "were",
    "at",
    "from",
}

_STOPWORDS = _FA_STOPWORDS | _EN_STOPWORDS


def _contains_farsi(text: str) -> bool:
    return bool(_FA_CHAR_PATTERN.search(text))


def _maybe_normalize(text: str) -> str:
    if settings.lexical_fa_preproc and _contains_farsi(text):
        return normalize_fa(text)
    return text


def _tokenize(text: str) -> set[str]:
    normalized = _maybe_normalize(text)
    tokens = {
        token.lower()
        for token in _TOKEN_PATTERN.findall(normalized)
        if token and token.lower() not in _STOPWORDS
    }
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return intersection / union


def overlap_score(query: str, text: str) -> float:
    """Compute lexical overlap between query and text."""
    query_tokens = _tokenize(query)
    text_tokens = _tokenize(text)
    return _jaccard(query_tokens, text_tokens)
