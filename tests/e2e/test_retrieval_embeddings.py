from __future__ import annotations

import re

import pytest

from embedding.adapter import EmbeddingAdapter


def _dot(left: list[float], right: list[float]) -> float:
    return sum(l * r for l, r in zip(left, right, strict=False))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", text.lower()))


@pytest.mark.integration
def test_mock_embedding_retrieval_multilingual(monkeypatch: pytest.MonkeyPatch) -> None:
    """EmbeddingAdapter mock should respect language-aware similarity."""

    monkeypatch.setenv("EMBED_MODEL_NAME", "mock")

    adapter = EmbeddingAdapter()

    chunks = ["کتاب ریاضی", "Math book", "History notes"]
    chunk_vectors = adapter.embed_texts(chunks)
    chunk_tokens = [_tokenize(chunk) for chunk in chunks]

    def score_chunks(query: str) -> dict[str, float]:
        query_vector = adapter.embed_texts([query])[0]
        query_tokens = _tokenize(query)
        scores: dict[str, float] = {}
        for chunk, vector, tokens in zip(chunks, chunk_vectors, chunk_tokens, strict=False):
            similarity = _dot(vector, query_vector)
            if query_tokens and tokens:
                common = len(query_tokens & tokens)
                total = len(query_tokens | tokens)
                overlap = common / total if total else 0.0
            else:
                overlap = 0.0
            scores[chunk] = 0.7 * similarity + 0.3 * overlap
        return scores

    fa_scores = score_chunks("ریاضی")
    en_scores = score_chunks("math")

    assert max(fa_scores, key=fa_scores.get) == "کتاب ریاضی"
    assert max(en_scores, key=en_scores.get) == "Math book"

    # Ensure cross-language matches score lower than same-language matches.
    assert fa_scores["کتاب ریاضی"] > fa_scores["Math book"]
    assert en_scores["Math book"] > en_scores["کتاب ریاضی"]
    assert en_scores["Math book"] > en_scores["History notes"]
    assert fa_scores["کتاب ریاضی"] > 0.0
    assert en_scores["Math book"] > 0.0
