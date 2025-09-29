"""Lightweight replacements for optional embedding backends."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .base import Embedder


class HashEmbedder(Embedder):
    """Deterministic hash-based embedder used as a graceful fallback.

    It mirrors the interface of heavier embedders while remaining dependency-free.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
                idx = int(digest[:16], 16) % self.dim
                vec[idx] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            vectors.append([v / norm for v in vec])
        return vectors
