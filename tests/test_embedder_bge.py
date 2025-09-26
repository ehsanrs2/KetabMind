import os
from collections.abc import Callable
from typing import Any, TypeVar, cast

import pytest

from core.embed import get_embedder

F = TypeVar("F", bound=Callable[..., Any])
slow = cast(Callable[[F], F], pytest.mark.slow)
parametrize_model_dim = cast(
    Callable[[F], F], pytest.mark.parametrize("model,dim", [("small", 384), ("base", 768)])
)


@slow
@parametrize_model_dim
def test_bge_embedder(monkeypatch: pytest.MonkeyPatch, model: str, dim: int) -> None:
    if os.getenv("EMBED_MODEL", "mock") == "mock":
        pytest.skip("real model tests require EMBED_MODEL!=mock")
    monkeypatch.setenv("EMBED_MODEL", model)
    embedder = get_embedder()
    vectors = embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == dim
