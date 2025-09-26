"""Smoke tests for the transformers-backed LLM."""

from __future__ import annotations

import os
from typing import Any

import pytest

from core.answer.llm import TransformersLLM

pytestmark = pytest.mark.slow


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Transformers smoke test skipped in CI")
def test_transformers_generate_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():  # pragma: no cover - requires GPU
        pytest.skip("CUDA is required for the transformers smoke test")

    class FakeBatch:
        def __init__(self, tensor: Any) -> None:
            self._data = {"input_ids": tensor}

        def to(self, device: str) -> FakeBatch:
            return self

        def items(self):
            return self._data.items()

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, prompt: str, return_tensors: str):
            tensor = torch.tensor([[1, 2]], dtype=torch.long)
            return FakeBatch(tensor)

        def decode(self, tokens: Any, skip_special_tokens: bool = True) -> str:
            return "stubbed answer"

    class FakeModel:
        def generate(self, **kwargs: Any):
            input_ids = kwargs["input_ids"]
            extension = torch.tensor([[3, 4]], dtype=torch.long)
            return torch.cat([input_ids, extension], dim=1)

    def fake_ensure(self: TransformersLLM) -> None:
        if self._model is not None:
            return
        self._device = "cuda"
        self._torch = torch
        self._tokenizer = FakeTokenizer()
        self._model = FakeModel()
        self._streamer_cls = transformers.TextIteratorStreamer

    monkeypatch.setattr(TransformersLLM, "_ensure_model", fake_ensure, raising=False)

    llm = TransformersLLM()
    result = llm.generate("Hello?")
    assert isinstance(result, str)
    assert result.strip()
