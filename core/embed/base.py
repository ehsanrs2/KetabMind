"""Embedding interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List


class Embedder(ABC):
    """Abstract embedder."""

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Return vector embeddings for given texts."""
        raise NotImplementedError
