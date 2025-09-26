"""Smoke tests for the Transformers LLM backend."""
from __future__ import annotations

import os
from typing import Any

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip GPU smoke test on CI"),
]


def _build_dummy_components(torch_module: Any) -> tuple[Any, Any]:
    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, Any]:
            _ = text
            tensor = torch_module.tensor([[1, 2]], dtype=torch_module.long)
            return {"input_ids": tensor, "attention_mask": torch_module.ones_like(tensor)}

        def decode(self, tokens: Any, skip_special_tokens: bool = True) -> str:
            _ = tokens, skip_special_tokens
            return "dummy completion"

    class DummyModel:
        def generate(self, **kwargs: Any) -> Any:
            input_ids = kwargs["input_ids"]
            extra = torch_module.full(
                (input_ids.size(0), 1),
                99,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            return torch_module.cat([input_ids, extra], dim=-1)

    return DummyTokenizer(), DummyModel()


def test_transformers_generate_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = pytest.importorskip("transformers")
    torch_module = pytest.importorskip("torch")

    if not torch_module.cuda.is_available():
        pytest.skip("CUDA device is required for this smoke test")

    tokenizer_stub, model_stub = _build_dummy_components(torch_module)

    monkeypatch.setenv("LLM_MODEL", "dummy-model")
    monkeypatch.setenv("LLM_DEVICE", "cuda")
    monkeypatch.setenv("LLM_LOAD_IN_4BIT", "false")

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer_stub,
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: model_stub,
    )

    from core.answer.llm import TransformersLLM

    llm = TransformersLLM()
    result = llm.generate("Hello", stream=False)

    assert isinstance(result, str)
    assert result
