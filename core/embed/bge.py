"""BGE embedder using sentence-transformers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from sentence_transformers import SentenceTransformer

from .base import Embedder

_MODEL_NAMES = {
    384: "BAAI/bge-small-en-v1.5",
    768: "BAAI/bge-base-en-v1.5",
}


class BgeEmbedder(Embedder):
    """Embedder backed by BGE models."""

    _model: ClassVar[SentenceTransformer | None] = None
    _model_name: ClassVar[str | None] = None

    def __init__(self, embed_dim: int = 384, model_name: str | None = None) -> None:
        if model_name is None:
            model_name = _MODEL_NAMES.get(embed_dim, _MODEL_NAMES[384])
        self.embed_dim = embed_dim
        self.model_name = model_name

        if BgeEmbedder._model is None or BgeEmbedder._model_name != model_name:
            BgeEmbedder._model = SentenceTransformer(model_name)
            BgeEmbedder._model_name = model_name

        self._model_instance = BgeEmbedder._model

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = self._model_instance.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, vec)) for vec in vectors]

    # Compatibility attribute
    @property
    def dim(self) -> int:  # pragma: no cover - simple property
        return self.embed_dim
