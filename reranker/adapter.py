"""Utilities for running cross-encoder rerankers."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

Pair = tuple[str, str]


class RerankerAdapter:
    """Adapter around HuggingFace cross-encoder rerankers."""

    def __init__(
        self,
        model_name: str,
        device: str | None,
        batch_size: int = 16,
        timeout_s: float = 10.0,
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.timeout_s = timeout_s

        resolved_device = device
        if resolved_device is None:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved_device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(self, pairs: Sequence[Pair]) -> list[float]:
        """Score (query, document) pairs with the cross-encoder."""
        if not pairs:
            return []

        scores: list[float] = []
        for batch in _batched(pairs, self.batch_size):
            queries = [q for q, _ in batch]
            docs = [d for _, d in batch]
            tokenized = self.tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            logits = self._infer_with_timeout(tokenized)
            logits = logits.to("cpu")
            batch_scores = self._logits_to_scores(logits)
            scores.extend(batch_scores)

        return scores

    def _run_inference(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.detach()

    def _infer_with_timeout(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        result: dict[str, torch.Tensor] = {}
        exception: dict[str, BaseException] = {}
        done = threading.Event()

        def target() -> None:
            try:
                result["value"] = self._run_inference(inputs)
            except BaseException as exc:  # noqa: BLE001
                exception["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        finished = done.wait(self.timeout_s)
        if not finished:
            raise TimeoutError(f"Reranker inference exceeded timeout of {self.timeout_s} seconds")
        thread.join()

        if "error" in exception:
            raise exception["error"]
        return result["value"]

    @staticmethod
    def _logits_to_scores(logits: torch.Tensor) -> list[float]:
        if logits.numel() == 0:
            return []
        if logits.shape[-1] == 1:
            probs = torch.sigmoid(logits.squeeze(-1))
        else:
            probs = torch.softmax(logits, dim=-1)[..., -1]
        return probs.detach().cpu().tolist()


def _batched(items: Sequence[Pair], batch_size: int) -> Iterable[Sequence[Pair]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
