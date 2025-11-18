from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

FloatArray: TypeAlias = NDArray[np.float32]

_MODEL_BY_NAME: dict[str, tuple[str, int]] = {
    "small": ("BAAI/bge-small-en-v1.5", 384),
    "base": ("BAAI/bge-base-en-v1.5", 768),
}


@dataclass
class EmbeddingAdapter:
    dim: int
    model_name: str
    _model_cache: ClassVar[dict[str, SentenceTransformer]] = {}
    _model: SentenceTransformer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.model_name not in self._model_cache:
            self._model_cache[self.model_name] = SentenceTransformer(self.model_name)
        self._model = self._model_cache[self.model_name]

    def embed_texts(self, texts: Iterable[str]) -> FloatArray:
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            text_list, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


class SmallEmbedding(EmbeddingAdapter):
    def __init__(self) -> None:
        model_name, dim = _MODEL_BY_NAME["small"]
        super().__init__(dim=dim, model_name=model_name)


class BaseEmbedding(EmbeddingAdapter):
    def __init__(self) -> None:
        model_name, dim = _MODEL_BY_NAME["base"]
        super().__init__(dim=dim, model_name=model_name)


def get_embedder(name: str) -> EmbeddingAdapter:
    key = name.lower()
    if key == "small":
        return SmallEmbedding()
    return BaseEmbedding()
