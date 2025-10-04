from __future__ import annotations

import types

import pytest

from utils import gpu_opt


def _mock_cuda(monkeypatch: pytest.MonkeyPatch, free_bytes: int, total_bytes: int) -> None:
    fake_cuda = types.SimpleNamespace()
    fake_cuda.is_available = lambda: True
    fake_cuda.mem_get_info = lambda index=0: (free_bytes, total_bytes)
    fake_cuda.get_device_properties = lambda index=0: types.SimpleNamespace(total_memory=total_bytes)
    fake_torch = types.SimpleNamespace(
        cuda=fake_cuda,
        bfloat16="bfloat16",
        float16="float16",
    )
    monkeypatch.setattr(gpu_opt, "torch", fake_torch, raising=False)
    monkeypatch.setattr(gpu_opt, "bnb", object(), raising=False)


def test_adjust_batch_size_downsizes_when_memory_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    free = 4 * 1024**3
    total = 8 * 1024**3
    _mock_cuda(monkeypatch, free_bytes=free, total_bytes=total)

    optimizer = gpu_opt.GPUOptimizer(device="cuda:0", safety_margin=1.0, reserved_memory_mb=0)
    adjusted = optimizer.adjust_batch_size(base_batch_size=16, per_sample_memory_bytes=512 * 1024**2)
    assert adjusted == 8


def test_trim_context_limits_tokens_based_on_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    free = 2 * 1024**3
    total = 8 * 1024**3
    _mock_cuda(monkeypatch, free_bytes=free, total_bytes=total)

    optimizer = gpu_opt.GPUOptimizer(device="cuda:0", safety_margin=1.0, reserved_memory_mb=0, tokens_per_gb=1024)
    tokens = list(range(5000))
    trimmed = optimizer.trim_context(tokens)
    assert len(trimmed) == 2048
    assert trimmed == tokens[-2048:]


def test_quantization_config_supports_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    free = total = 8 * 1024**3
    _mock_cuda(monkeypatch, free_bytes=free, total_bytes=total)

    config8 = gpu_opt.quantization_config("8bit")
    assert config8 == {"load_in_8bit": True}

    config4 = gpu_opt.quantization_config("4bit")
    assert config4["load_in_4bit"] is True
    assert config4["bnb_4bit_use_double_quant"] is True


def test_adjust_batch_size_returns_base_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda = types.SimpleNamespace()
    fake_cuda.is_available = lambda: False
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr(gpu_opt, "torch", fake_torch, raising=False)

    optimizer = gpu_opt.GPUOptimizer(device="cpu")
    adjusted = optimizer.adjust_batch_size(base_batch_size=4, per_sample_memory_bytes=256 * 1024**2)
    assert adjusted == 4
