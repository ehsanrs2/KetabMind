from __future__ import annotations

import re

import pytest

from lexical.overlap import _STOPWORDS, overlap_score

_TOKEN_PATTERN = re.compile(r"[\w\u200c]+", re.UNICODE)


def _overlap_without_normalization(query: str, text: str) -> float:
    def _tokenize(raw: str) -> set[str]:
        tokens = {
            token.lower()
            for token in _TOKEN_PATTERN.findall(raw)
            if token and token.lower() not in _STOPWORDS
        }
        return tokens

    query_tokens = _tokenize(query)
    text_tokens = _tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0
    intersection = len(query_tokens & text_tokens)
    union = len(query_tokens | text_tokens)
    return intersection / union if union else 0.0


def test_overlap_persian_only() -> None:
    query = "کتاب جدید و جذاب"
    text = "این کتاب جذاب است و کتاب جذاب ماندگار است"
    score = overlap_score(query, text)
    assert pytest.approx(score, rel=0.0) == 2 / 4


def test_overlap_english_only() -> None:
    query = "deep learning models"
    text = "This article covers deep learning and neural networks"
    score = overlap_score(query, text)
    assert pytest.approx(score, rel=0.0) == 2 / 8


def test_overlap_mixed_text() -> None:
    query = "کتاب deep learning"
    text = "این کتاب درباره deep learning و شبکه های عصبی است"
    score = overlap_score(query, text)
    assert score > 0.0
    assert score < 1.0


def test_normalization_improves_fa_overlap() -> None:
    query = "كتاب‌ها"
    text = "کتاب‌ها"
    raw_score = _overlap_without_normalization(query, text)
    normalized_score = overlap_score(query, text)
    assert raw_score == 0.0
    assert normalized_score > raw_score
    assert normalized_score == pytest.approx(1.0)
