from __future__ import annotations

from core.config import settings

from .base import Embedder
from .mock import MockEmbedder


def get_embedder() -> Embedder:
    """Return an embedder based on settings.embed_model.

    Supported values:
    - mock: lightweight deterministic mock (small dim)
    - small: BGE small (384)
    - base: BGE base (768)
    """
    name = (settings.embed_model or "mock").lower()
    if name == "mock":
        return MockEmbedder()
    # Lazy import to avoid heavy deps unless requested
    try:
        from .bge import BgeEmbedder
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "BGE requires sentence-transformers. Install it or set EMBED_MODEL=mock."
        ) from exc
    if name == "base":
        return BgeEmbedder(embed_dim=768)
    # default to small
    return BgeEmbedder(embed_dim=384)
