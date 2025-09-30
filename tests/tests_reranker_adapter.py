import time

import pytest
import torch

from reranker.adapter import RerankerAdapter


class DummyModel:
    def __init__(self) -> None:
        self.to_calls = []
        self.eval_called = False

    def to(self, device: torch.device) -> "DummyModel":
        self.to_calls.append(device)
        return self

    def eval(self) -> "DummyModel":
        self.eval_called = True
        return self

    def __call__(self, **kwargs):  # pragma: no cover - replaced in tests
        raise NotImplementedError


class DummyTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, queries, docs, **kwargs):
        self.calls.append((tuple(queries), tuple(docs)))
        batch_size = len(queries)
        return {
            "input_ids": torch.zeros((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }


@pytest.fixture()
def patched_hf(monkeypatch):
    tokenizer = DummyTokenizer()
    model = DummyModel()

    monkeypatch.setattr(
        "reranker.adapter.AutoTokenizer.from_pretrained", lambda *_, **__: tokenizer
    )
    monkeypatch.setattr(
        "reranker.adapter.AutoModelForSequenceClassification.from_pretrained",
        lambda *_, **__: model,
    )
    return tokenizer, model


def test_score_pairs_batches_and_returns_scores(monkeypatch, patched_hf):
    tokenizer, _ = patched_hf

    adapter = RerankerAdapter("dummy", device="cpu", batch_size=2, timeout_s=1.0)

    logits_batches = [
        torch.tensor([[0.1, 0.9], [0.9, 0.1]]),
        torch.tensor([[0.2, 0.8], [0.5, 0.5]]),
        torch.tensor([[1.0, 0.0]]),
    ]

    def fake_infer(_inputs):
        return logits_batches.pop(0)

    monkeypatch.setattr(adapter, "_infer_with_timeout", fake_infer)

    pairs = [(f"q{i}", f"d{i}") for i in range(5)]
    scores = adapter.score_pairs(pairs)

    assert len(scores) == 5
    assert pytest.approx(scores, rel=1e-5) == [
        0.689974,
        0.310026,
        0.645656,
        0.5,
        0.268941,
    ]
    assert len(tokenizer.calls) == 3
    assert all(len(qs) <= 2 for qs, _ in tokenizer.calls)


def test_device_selection(monkeypatch, patched_hf):
    _, model = patched_hf

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    adapter = RerankerAdapter("dummy", device=None)

    assert adapter.device.type == "cuda"
    assert model.to_calls[-1].type == "cuda"

    adapter_cpu = RerankerAdapter("dummy", device="cpu")
    assert adapter_cpu.device.type == "cpu"
    assert model.to_calls[-1].type == "cpu"


def test_timeout(monkeypatch, patched_hf):
    adapter = RerankerAdapter("dummy", device="cpu", timeout_s=0.1)

    def slow_run(_inputs):
        time.sleep(0.2)
        return torch.tensor([[0.0, 1.0]])

    monkeypatch.setattr(adapter, "_run_inference", slow_run)

    with pytest.raises(TimeoutError):
        adapter.score_pairs([("q", "d")])
