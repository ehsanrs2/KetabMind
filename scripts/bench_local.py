"""Local benchmarking utility for embeddings and generation latency."""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency for GPU metrics
    import torch
except ImportError:  # pragma: no cover - allow running without torch installed
    torch = None  # type: ignore[assignment]

from core.answer.llm import LLM, get_llm
from embedding.adapter import EmbeddingAdapter


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkArgs:
    """Container holding parsed CLI arguments."""

    embed_model: str
    batch_size: int
    sample_text: str
    prompt: str
    repeats: int
    warmup: int
    device: str | None
    output_path: Path


def _parse_args() -> BenchmarkArgs:
    parser = argparse.ArgumentParser(description="Benchmark local embedding and generation speed")
    parser.add_argument("--embed-model", default=os.getenv("EMBED_MODEL_NAME", "bge-m3"))
    parser.add_argument("--batch-size", type=int, default=16, help="Initial embedding batch size")
    parser.add_argument(
        "--sample-text",
        default="KetabMind benchmarking text",
        help="Sample text duplicated to form embedding batches",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Question: What does KetabMind benchmark?\n"
            "Contexts:\n1. (doc:1) Sample context for benchmarking.\nAnswer:"
        ),
        help="Prompt sent to the configured language model",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations before timing")
    parser.add_argument("--device", default=None, help="Preferred device for embeddings (e.g. cuda:0)")
    parser.add_argument(
        "--output",
        default=str(Path.home() / ".ketabmind" / "benchmarks.json"),
        help="Destination file storing benchmark history",
    )
    args = parser.parse_args()
    return BenchmarkArgs(
        embed_model=args.embed_model,
        batch_size=max(args.batch_size, 1),
        sample_text=args.sample_text,
        prompt=args.prompt,
        repeats=max(args.repeats, 1),
        warmup=max(args.warmup, 0),
        device=args.device,
        output_path=Path(args.output).expanduser(),
    )


def _torch_cuda_available(device: str | None = None) -> bool:
    if torch is None:
        return False
    if not hasattr(torch, "cuda"):
        return False
    cuda = torch.cuda
    if not callable(getattr(cuda, "is_available", None)):
        return False
    if not cuda.is_available():  # type: ignore[no-any-return]
        return False
    if device is None:
        return True
    return device.startswith("cuda")


def _reset_gpu_stats(device: str | None = None) -> None:
    if not _torch_cuda_available(device):
        return
    cuda = torch.cuda
    if hasattr(cuda, "empty_cache"):
        cuda.empty_cache()
    if hasattr(cuda, "reset_peak_memory_stats"):
        try:
            cuda.reset_peak_memory_stats()  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover - compatibility for older torch
            cuda.reset_peak_memory_stats()  # type: ignore[misc]


def _max_gpu_memory(device: str | None = None) -> int | None:
    if not _torch_cuda_available(device):
        return None
    cuda = torch.cuda
    if not hasattr(cuda, "max_memory_allocated"):
        return None
    try:
        return int(cuda.max_memory_allocated())  # type: ignore[no-any-return]
    except TypeError:  # pragma: no cover - compatibility guard
        return None


def _synchronize_if_needed(device: str | None = None) -> None:
    if not _torch_cuda_available(device):
        return
    cuda = torch.cuda
    sync = getattr(cuda, "synchronize", None)
    if callable(sync):
        sync()


def _time_operation(
    operation: Callable[[], Any],
    repeats: int,
    warmup: int,
    device: str | None = None,
) -> list[float]:
    for _ in range(warmup):
        operation()
        _synchronize_if_needed(device)

    latencies: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        _synchronize_if_needed(device)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def auto_tune_batch_size(
    factory: Callable[[int], EmbeddingAdapter],
    start_batch: int,
    sample_text: str,
) -> int:
    batch_size = max(start_batch, 1)
    while batch_size >= 1:
        adapter = factory(batch_size)
        texts = [sample_text] * batch_size
        try:
            adapter.embed_texts(texts, batch_size=batch_size)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message and "cuda error" not in message:
                raise
            logger.warning("Batch size %d failed due to OOM; retrying with half", batch_size)
            batch_size //= 2
            if _torch_cuda_available(adapter.device):
                torch.cuda.empty_cache()
            if batch_size < 1:
                raise RuntimeError("Unable to find a valid batch size due to OOM") from exc
            continue
        return batch_size
    raise RuntimeError("Batch size tuning exhausted all options")


def measure_embedding_performance(
    adapter: EmbeddingAdapter,
    sample_text: str,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    texts = [sample_text] * adapter.batch_size
    _reset_gpu_stats(adapter.device)

    def _embed() -> None:
        adapter.embed_texts(texts, batch_size=len(texts))

    latencies = _time_operation(_embed, repeats=repeats, warmup=warmup, device=adapter.device)
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    throughput = 0.0
    if avg_latency > 0:
        throughput = len(texts) / (avg_latency / 1000)

    return {
        "batch_size": len(texts),
        "latencies_ms": latencies,
        "throughput_per_second": throughput,
        "gpu_memory_bytes": _max_gpu_memory(adapter.device),
    }


def _consume_output(output: str | Iterable[str]) -> str:
    if isinstance(output, str):
        return output
    return "".join(segment for segment in output)


def measure_generation_performance(
    llm: LLM,
    prompt: str,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    device = None
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[no-any-return]
        device = "cuda"

    def _generate() -> str:
        output = llm.generate(prompt)
        return _consume_output(output)

    _reset_gpu_stats(device)
    latencies = _time_operation(lambda: _generate(), repeats=repeats, warmup=warmup, device=device)
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    return {
        "latencies_ms": latencies,
        "avg_latency_ms": avg_latency,
        "gpu_memory_bytes": _max_gpu_memory(device),
    }


def _load_existing_benchmarks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return []
    if not content.strip():
        return []
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def _write_results(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history = _load_existing_benchmarks(path)
    history.append(record)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def run_benchmark(args: BenchmarkArgs) -> dict[str, Any]:
    def _create_adapter(batch_size: int) -> EmbeddingAdapter:
        return EmbeddingAdapter(
            model_name=args.embed_model,
            batch_size=batch_size,
            device=args.device,
        )

    tuned_batch = auto_tune_batch_size(_create_adapter, args.batch_size, args.sample_text)
    adapter = _create_adapter(tuned_batch)
    embedding_metrics = measure_embedding_performance(
        adapter,
        sample_text=args.sample_text,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    llm = get_llm()
    generation_metrics = measure_generation_performance(
        llm,
        prompt=args.prompt,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embed_model": args.embed_model,
        "batch_size": tuned_batch,
        "embedding": embedding_metrics,
        "generation": generation_metrics,
    }
    _write_results(args.output_path, record)
    return record


def main() -> None:  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = _parse_args()
    record = run_benchmark(args)
    logger.info("Benchmark recorded: batch=%d, embed throughput=%.2f/s", record["batch_size"], record["embedding"].get("throughput_per_second", 0.0))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
