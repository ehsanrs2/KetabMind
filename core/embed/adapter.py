from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingAdapter:
    dim: int

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError


class SmallEmbedding(EmbeddingAdapter):
    def __init__(self) -> None:
        super().__init__(dim=64)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        return _hash_embed(texts, self.dim)


class BaseEmbedding(EmbeddingAdapter):
    def __init__(self) -> None:
        super().__init__(dim=384)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        return _hash_embed(texts, self.dim)


def _hash_embed(texts: Iterable[str], dim: int) -> np.ndarray:
    vecs: list[np.ndarray] = []
    for t in texts:
        v = np.zeros(dim, dtype=np.float32)
        for tok in t.lower().split():
            h = hashlib.sha256(tok.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % dim
            v[idx] += 1.0
        norm = np.linalg.norm(v) or 1.0
        vecs.append(v / norm)
    return np.stack(vecs, axis=0) if vecs else np.zeros((0, dim), dtype=np.float32)


def get_embedder(name: str) -> EmbeddingAdapter:
    if name.lower() == "small":
        return SmallEmbedding()
    return BaseEmbedding()
