"""Embedding adapter using SentenceTransformers."""
from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer  # type: ignore

FloatArray: TypeAlias = NDArray[np.float32]

@dataclass
class EmbeddingAdapter:
    dim: int
    def embed(self, texts: Iterable[str]) -> FloatArray:
        raise NotImplementedError

class SmallEmbedding(EmbeddingAdapter):
    """Mock embedding for testing (fast, random)."""
    def __init__(self) -> None:
        super().__init__(dim=64)
    
    def embed(self, texts: Iterable[str]) -> FloatArray:
        return np.random.rand(len(list(texts)), self.dim).astype(np.float32)

class BaseEmbedding(EmbeddingAdapter):
    """Real semantic embedding using BGE model."""
    def __init__(self) -> None:
        # BGE-base dimensions (768)
        super().__init__(dim=768)
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    def embed(self, texts: Iterable[str]) -> FloatArray:
        # تولید بردارهای واقعی
        embeddings = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

def get_embedder(name: str) -> EmbeddingAdapter:
    if name.lower() == "small":
        return SmallEmbedding()
    return BaseEmbedding()