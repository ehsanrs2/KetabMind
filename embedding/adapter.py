"""Embedding adapter for supported multilingual models."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable

import structlog

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in tests
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover - optional dependency in tests
    AutoModel = AutoTokenizer = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


logger = structlog.get_logger(__name__)


class EmbeddingAdapter:
    """Adapter that wraps supported embedding models.

    The adapter supports loading SentenceTransformer models when available,
    falling back to a manual ``AutoModel`` + ``AutoTokenizer`` pipeline.
    """

    SUPPORTED_MODELS = {
        "bge-m3",
        "intfloat/multilingual-e5-base",
        "mock",
    }

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        env_model = os.getenv("EMBED_MODEL_NAME")
        env_batch = os.getenv("BATCH_SIZE")

        self.model_name = model_name or env_model or "bge-m3"
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{self.model_name}' is not supported. Supported models: {sorted(self.SUPPORTED_MODELS)}"
            )

        self.is_mock = self.model_name == "mock"

        if batch_size is not None:
            self.batch_size = batch_size
        elif env_batch:
            try:
                self.batch_size = int(env_batch)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError("BATCH_SIZE environment variable must be an integer") from exc
        else:
            self.batch_size = 16

        if not self.is_mock and torch is None:
            raise ImportError(f"PyTorch is required for embedding model '{self.model_name}'.")

        if not self.is_mock and AutoModel is None:
            raise ImportError(f"transformers is required for embedding model '{self.model_name}'.")

        if device is not None:
            requested_device = device
        else:
            cuda_available = bool(torch and torch.cuda.is_available())
            requested_device = "cuda" if cuda_available else "cpu"

        cuda_available = bool(torch and torch.cuda.is_available())
        if requested_device.startswith("cuda") and not cuda_available:
            logger.warning("CUDA requested for embeddings but not available; falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = "cpu" if self.is_mock else requested_device

        if self.is_mock:
            # Mock embeddings are CPU-only and do not require further setup.
            self.quantization_mode = None
            self.model = None
            self.tokenizer = None
            self.uses_sentence_transformer = False
            return

        self.uses_sentence_transformer = False
        self.model = None
        self.tokenizer = None

        quant_env = (os.getenv("EMBED_QUANT") or "").strip().lower()
        if quant_env in {"8", "8bit"}:
            self.quantization_mode: str | None = "8bit"
        elif quant_env in {"4", "4bit"}:
            self.quantization_mode = "4bit"
        elif quant_env:
            raise ValueError("EMBED_QUANT must be empty, '8bit', or '4bit'")
        else:
            self.quantization_mode = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model using the appropriate backend."""
        if self.quantization_mode:
            self._load_quantized_model()
            return

        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.uses_sentence_transformer = True
                return
            except Exception:  # pragma: no cover - fallback when loading fails
                # Fall back to AutoModel pipeline if instantiation fails for any reason.
                self.uses_sentence_transformer = False

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_kwargs = {}
        if "cuda" in self.device:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        try:
            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        except (ValueError, RuntimeError):  # pragma: no cover - fallback for unsupported fp16
            if model_kwargs.get("torch_dtype") == torch.float16:
                model_kwargs["torch_dtype"] = torch.float32
                self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
            else:
                raise

        self.model.to(self.device)
        self.model.eval()

    def _load_quantized_model(self) -> None:
        """Load a quantized model using bitsandbytes when requested."""
        assert self.quantization_mode is not None

        try:
            import bitsandbytes  # type: ignore  # noqa: F401
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise ImportError("bitsandbytes is required when EMBED_QUANT is set.") from exc

        from transformers import BitsAndBytesConfig

        if self.quantization_mode == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
        self.model.eval()

    @staticmethod
    def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        masked_embeddings = token_embeddings * mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        mask_sum = mask.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / mask_sum

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Embed a batch of texts and return list of embeddings."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Iterable):
            raise TypeError("texts must be an iterable of strings")

        if not texts:
            return []

        if self.is_mock:
            return [self._hash_embed(text) for text in texts]

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

        results: list[list[float]] = []
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

    @staticmethod
    def _hash_embed(text: str, dim: int = 64) -> list[float]:
        """Deterministic hash-based embedding used for mock mode."""
        vec = [0.0] * dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]
