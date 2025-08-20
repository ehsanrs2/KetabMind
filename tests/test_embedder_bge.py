import os

import pytest

from core.embed import get_embedder


@pytest.mark.parametrize("model,dim", [("small", 384), ("base", 768)])  # type: ignore[misc]
@pytest.mark.slow  # type: ignore[misc]
def test_bge_embedder(monkeypatch: pytest.MonkeyPatch, model: str, dim: int) -> None:
    if os.getenv("EMBED_MODEL", "mock") == "mock":
        pytest.skip("real model tests require EMBED_MODEL!=mock")
    monkeypatch.setenv("EMBED_MODEL", model)
    embedder = get_embedder()
    vectors = embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == dim
