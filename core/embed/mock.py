"""Mock embedder returning zero vectors."""

from collections.abc import Iterable
from typing import List

from .base import Embedder


class MockEmbedder(Embedder):
    """Mock embedder used for testing."""

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        return [[0.0] * 3 for _ in texts]
