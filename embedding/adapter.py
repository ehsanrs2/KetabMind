"""Embedding adapter for supported multilingual models."""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


class EmbeddingAdapter:
    """Adapter that wraps supported embedding models.

    The adapter supports loading SentenceTransformer models when available,
    falling back to a manual ``AutoModel`` + ``AutoTokenizer`` pipeline.
    """

    SUPPORTED_MODELS = {
        "bge-m3",
        "intfloat/multilingual-e5-base",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        env_model = os.getenv("EMBED_MODEL_NAME")
        env_batch = os.getenv("BATCH_SIZE")

        self.model_name = model_name or env_model or "bge-m3"
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{self.model_name}' is not supported. Supported models: {sorted(self.SUPPORTED_MODELS)}"
            )

        if batch_size is not None:
            self.batch_size = batch_size
        elif env_batch:
            try:
                self.batch_size = int(env_batch)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError("BATCH_SIZE environment variable must be an integer") from exc
        else:
            self.batch_size = 16

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.uses_sentence_transformer = False
        self.model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model using the appropriate backend."""
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.uses_sentence_transformer = True
                return
            except Exception:  # pragma: no cover - fallback when loading fails
                # Fall back to AutoModel pipeline if instantiation fails for any reason.
                self.uses_sentence_transformer = False

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        masked_embeddings = token_embeddings * mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        mask_sum = mask.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / mask_sum

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Embed a batch of texts and return list of embeddings."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Iterable):
            raise TypeError("texts must be an iterable of strings")

        if not texts:
            return []

        effective_batch_size = (
            self.batch_size if batch_size == 16 and self.batch_size != 16 else batch_size
        )

        if self.uses_sentence_transformer:
            embeddings = self.model.encode(  # type: ignore[union-attr]
                texts,
                batch_size=effective_batch_size,
                convert_to_numpy=True,
                device=self.device,
                show_progress_bar=False,
            )
            return embeddings.tolist()

        assert self.model is not None and self.tokenizer is not None

        results: List[List[float]] = []
        for start in range(0, len(texts), effective_batch_size):
            batch_texts = texts[start : start + effective_batch_size]
            encodings = self.tokenizer(  # type: ignore[operator]
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = self.model(**encodings)  # type: ignore[misc]
                token_embeddings = outputs.last_hidden_state
                pooled = self._mean_pooling(token_embeddings, encodings["attention_mask"])
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

            results.extend(normalized.cpu().tolist())

        return results
