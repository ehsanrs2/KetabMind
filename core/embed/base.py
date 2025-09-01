"""Embedding interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterable


class Embedder(ABC):
    """Abstract embedder."""

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Return vector embeddings for given texts."""
        raise NotImplementedError
