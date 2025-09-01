"""Mock embedder returning simple hash-based vectors."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .base import Embedder


class MockEmbedder(Embedder):
    """Mock embedder used for testing and CI.

    - Deterministic bag-of-words style hashing into a small fixed dim.
    - Lightweight and offline; suitable for unit tests.
    """

    dim: int = 64

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in t.lower().split():
                h = hashlib.sha256(tok.encode("utf-8")).hexdigest()
                idx = int(h[:8], 16) % self.dim
                vec[idx] += 1.0
            # l2 normalize
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out
