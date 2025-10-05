"""Tests for the local benchmarking utility."""

from __future__ import annotations

import types

import pytest

from scripts import bench_local


class _DummyAdapter:
    def __init__(self, batch_size: int, fail_threshold: int | None = None) -> None:
        self.batch_size = batch_size
        self.device = "cuda"
        self._fail_threshold = fail_threshold

    def embed_texts(self, texts: list[str], batch_size: int) -> None:
        if self._fail_threshold is not None and batch_size > self._fail_threshold:
            raise RuntimeError("CUDA out of memory")


class _DummyLLM:
    def __init__(self) -> None:
        self.invocations = 0

    def generate(self, prompt: str, stream: bool = False):  # noqa: D401 - interface required
        self.invocations += 1
        return "response"


@pytest.fixture(autouse=True)
def _fake_torch(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    class _FakeCuda:
        def __init__(self) -> None:
            self._available = True

        def is_available(self) -> bool:  # noqa: D401 - mimic torch API
            return True

        def empty_cache(self) -> None:  # pragma: no cover - no-op
            return None

        def reset_peak_memory_stats(self) -> None:  # pragma: no cover - no-op
            return None

        def synchronize(self) -> None:  # pragma: no cover - no-op
            return None

        def max_memory_allocated(self) -> int:
            return 2048

    fake_torch = types.SimpleNamespace(cuda=_FakeCuda())
    monkeypatch.setattr(bench_local, "torch", fake_torch)
    return fake_torch


def test_auto_tune_batch_size_reduces_on_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    def _factory(batch: int) -> _DummyAdapter:
        return _DummyAdapter(batch, fail_threshold=4)

    tuned = bench_local.auto_tune_batch_size(_factory, start_batch=8, sample_text="x")
    assert tuned == 4


def test_measure_embedding_records_gpu_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _DummyAdapter(batch_size=2)

    metrics = bench_local.measure_embedding_performance(adapter, sample_text="x", repeats=1, warmup=0)

    assert metrics["batch_size"] == 2
    assert metrics["gpu_memory_bytes"] == 2048


def test_measure_generation_uses_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_llm = _DummyLLM()
    monkeypatch.setattr(bench_local, "get_llm", lambda: dummy_llm)

    metrics = bench_local.measure_generation_performance(dummy_llm, prompt="hello", repeats=1, warmup=0)

    assert metrics["avg_latency_ms"] >= 0.0
    assert dummy_llm.invocations >= 1
