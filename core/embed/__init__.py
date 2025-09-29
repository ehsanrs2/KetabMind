from __future__ import annotations

from core import config

from .base import Embedder
from .fallback import HashEmbedder
from .mock import MockEmbedder


def _ensure_settings() -> str:
    return (config.get_settings().embed_model or "mock").lower()


def _make_bge(dim: int) -> Embedder:
    try:
        from .bge import BgeEmbedder
    except Exception:  # pragma: no cover - gracefully fall back when deps missing
        return HashEmbedder(dim)
    try:
        return BgeEmbedder(embed_dim=dim)
    except Exception:  # pragma: no cover - runtime model issues
        return HashEmbedder(dim)


def get_embedder() -> Embedder:
    """Return an embedder based on configuration.

    Supported values:
    - mock: lightweight deterministic mock (small dim)
    - small: BGE small (384) or fallback hash-based equivalent when deps unavailable
    - base: BGE base (768) or fallback hash-based equivalent when deps unavailable
    """

    name = _ensure_settings()
    if name == "mock":
        return MockEmbedder()
    if name == "base":
        return _make_bge(768)
    return _make_bge(384)
