"""Embedding factory."""

from __future__ import annotations

import os

from .base import Embedder
from .bge import BgeEmbedder
from .mock import MockEmbedder


def get_embedder() -> Embedder:
    """Return embedder based on `EMBED_MODEL` env variable."""
    model = os.getenv("EMBED_MODEL", "small").lower()
    if model == "mock":
        return MockEmbedder()
    if model == "base":
        return BgeEmbedder(embed_dim=768)
    if model == "small":
        return BgeEmbedder(embed_dim=384)
    raise ValueError(f"Unsupported EMBED_MODEL: {model}")
