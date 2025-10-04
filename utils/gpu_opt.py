"""GPU optimization helpers for dynamic batching and quantization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, TypeVar

import structlog

try:  # pragma: no cover - optional dependency in some environments
    import torch
except ImportError:  # pragma: no cover - handled gracefully in helpers
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import bitsandbytes as bnb  # noqa: F401
except ImportError:  # pragma: no cover - handled gracefully in helpers
    bnb = None  # type: ignore[assignment]


logger = structlog.get_logger(__name__)

T = TypeVar("T")


def _parse_device_index(device: str | int | None) -> int:
    if isinstance(device, int):
        return device
    if not device:
        return 0
    if isinstance(device, str) and ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except ValueError:  # pragma: no cover - defensive guard
            logger.warning("Unable to parse CUDA device index from %s; defaulting to 0", device)
            return 0
    return 0


@dataclass(slots=True)
class GPUOptimizer:
    """Utility class that tunes GPU memory usage and batching."""

    device: str | int | None = None
    safety_margin: float = 0.9
    reserved_memory_mb: int = 512
    tokens_per_gb: int = 4096
    _device_index: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.device is None:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        self._device_index = _parse_device_index(self.device if not isinstance(self.device, int) else self.device)

    @property
    def reserved_memory_bytes(self) -> int:
        return int(self.reserved_memory_mb * 1024**2)

    def _is_cuda(self) -> bool:
        return (
            isinstance(self.device, (str, int))
            and torch is not None
            and hasattr(torch, "cuda")
            and torch.cuda.is_available()
            and str(self.device).startswith("cuda")
        )

    def get_total_memory(self) -> int:
        if not self._is_cuda():
            return 0
        try:
            props = torch.cuda.get_device_properties(self._device_index)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to read CUDA device properties.")
            return 0
        return int(getattr(props, "total_memory", 0))

    def get_available_memory(self) -> int:
        total = self.get_total_memory()
        if total <= 0 or not self._is_cuda():
            return 0

        free = total
        if hasattr(torch.cuda, "mem_get_info"):
            try:
                free, _total = torch.cuda.mem_get_info(self._device_index)
                if _total > 0:
                    total = int(_total)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to query CUDA memory info; using total memory as proxy.")
                free = total

        limit = int(total * self.safety_margin)
        usable = min(free, limit) - self.reserved_memory_bytes
        return max(usable, 0)

    def adjust_batch_size(
        self,
        base_batch_size: int,
        per_sample_memory_bytes: int,
        *,
        min_batch_size: int = 1,
    ) -> int:
        if base_batch_size <= 0:
            raise ValueError("base_batch_size must be positive")
        if per_sample_memory_bytes <= 0:
            raise ValueError("per_sample_memory_bytes must be positive")

        available = self.get_available_memory()
        if available <= 0:
            return max(base_batch_size, min_batch_size)

        max_fit = max(available // per_sample_memory_bytes, 1)
        adjusted = min(base_batch_size, int(max_fit))
        return max(adjusted, min_batch_size)

    def max_context_tokens(self, explicit_max: int | None = None) -> int | None:
        if explicit_max is not None:
            return max(explicit_max, 0)
        available = self.get_available_memory()
        if available <= 0:
            return None
        tokens = int((available / (1024**3)) * self.tokens_per_gb)
        return tokens if tokens > 0 else None

    def trim_context(self, tokens: Sequence[T], max_tokens: int | None = None) -> Sequence[T]:
        limit = self.max_context_tokens(max_tokens)
        if limit is None or len(tokens) <= limit:
            return tokens
        if isinstance(tokens, str):
            return tokens[-limit:]
        if isinstance(tokens, tuple):
            return tokens[-limit:]
        return tokens[-limit:]


def quantization_config(mode: str | None) -> dict[str, object]:
    if not mode:
        return {}
    normalized = mode.lower()
    if normalized not in {"4bit", "8bit"}:
        raise ValueError("Quantization mode must be '4bit' or '8bit'")
    if bnb is None:
        raise ImportError("bitsandbytes is required for quantization support")

    if normalized == "8bit":
        return {"load_in_8bit": True}

    compute_dtype: object
    if torch is not None and hasattr(torch, "bfloat16"):
        compute_dtype = torch.bfloat16
    elif torch is not None and hasattr(torch, "float16"):
        compute_dtype = torch.float16
    else:  # pragma: no cover - torch missing in runtime envs
        compute_dtype = "float16"

    return {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": compute_dtype,
    }


def estimate_context(tokens: Iterable[T], optimizer: GPUOptimizer, max_tokens: int | None = None) -> Sequence[T]:
    token_sequence = list(tokens) if not isinstance(tokens, (list, tuple, str)) else tokens
    trimmed = optimizer.trim_context(token_sequence, max_tokens=max_tokens)
    return trimmed
