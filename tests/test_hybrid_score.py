import math

import pytest

from scoring.hybrid import hybrid_score, parse_weights


def test_parse_weights_basic():
    spec = "cosine=0.5, lexical=0.3, reranker=0.2"
    assert parse_weights(spec) == {
        "cosine": 0.5,
        "lexical": 0.3,
        "reranker": 0.2,
    }


def test_parse_weights_invalid_entry():
    with pytest.raises(ValueError):
        parse_weights("cosine")


def test_parse_weights_empty_string():
    assert parse_weights("") == {}


def test_hybrid_score_normalizes_weights():
    weights = parse_weights("cosine=2, lexical=1")
    score = hybrid_score(0.6, 0.3, None, weights)
    assert math.isclose(score, 0.5)


def test_hybrid_score_missing_components_treated_as_zero():
    weights = {"cosine": 1.0, "lexical": 1.0}
    score = hybrid_score(0.8, None, None, weights)
    assert math.isclose(score, 0.4)


def test_hybrid_score_zero_total_weight_returns_zero():
    score = hybrid_score(0.9, 0.9, 0.9, {"cosine": 0.0})
    assert score == 0.0


def test_hybrid_score_ignores_unknown_weights():
    weights = {"cosine": 1.0, "unknown": 100.0}
    score = hybrid_score(0.2, None, None, weights)
    assert math.isclose(score, 0.2)
