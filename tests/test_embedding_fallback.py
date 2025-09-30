"""Tests for embedding adapter fallback behaviors."""

from __future__ import annotations

import os

import pytest

from embedding.adapter import EmbeddingAdapter


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch: pytest.MonkeyPatch) -> None:
    original = os.environ.get("EMBED_MODEL_NAME")
    yield
    if original is None:
        monkeypatch.delenv("EMBED_MODEL_NAME", raising=False)
    else:
        monkeypatch.setenv("EMBED_MODEL_NAME", original)


def test_mock_embedding_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """When EMBED_MODEL_NAME=mock the adapter should return hash-based vectors."""
    monkeypatch.setenv("EMBED_MODEL_NAME", "mock")

    adapter = EmbeddingAdapter()

    text = "Hello world"
    vec1 = adapter.embed_texts([text])[0]
    vec2 = adapter.embed_texts([text])[0]

    assert vec1 == vec2
    assert len(vec1) == 64

    other_vec = adapter.embed_texts(["A different sentence"])[0]
    assert other_vec != vec1
