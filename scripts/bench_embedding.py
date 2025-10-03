"""Benchmark helper for embedding adapters."""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import time

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

import torch

from embedding.adapter import EmbeddingAdapter


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s: %(message)s")


def _collect_cpu_memory() -> float | None:
    if psutil is None:  # pragma: no cover - optional dependency
        return None

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)


def _collect_gpu_memory(device: str) -> float | None:
    if "cuda" not in device or not torch.cuda.is_available():
        return None

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark embedding latency and memory usage.")
    parser.add_argument("--model-name", default=os.getenv("EMBED_MODEL_NAME", "bge-m3"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed runs to average.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations.")
    parser.add_argument("--device", default=None, help="Force device (e.g., cuda or cpu).")
    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        default=os.getenv("EMBED_QUANT", "none") or "none",
    )
    parser.add_argument(
        "--sample-text",
        default="KetabMind embedding benchmark sample text.",
        help="Text used to generate the benchmark batch.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    if args.quantization != "none":
        os.environ["EMBED_QUANT"] = args.quantization
    elif "EMBED_QUANT" in os.environ:
        del os.environ["EMBED_QUANT"]

    os.environ["EMBED_MODEL_NAME"] = args.model_name

    texts: list[str] = [args.sample_text for _ in range(args.batch_size)]

    adapter = EmbeddingAdapter(batch_size=args.batch_size, device=args.device)
    device = adapter.device

    logging.info(
        "Benchmarking model=%s device=%s quant=%s", args.model_name, device, args.quantization
    )

    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(args.warmup):
        adapter.embed_texts(texts, batch_size=args.batch_size)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        adapter.embed_texts(texts, batch_size=args.batch_size)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000)

    cpu_memory = _collect_cpu_memory()
    gpu_memory = _collect_gpu_memory(device)

    logging.info(
        "Latency ms -> avg: %.2f | p50: %.2f | p95: %.2f | max: %.2f",
        statistics.mean(latencies_ms),
        statistics.median(latencies_ms),
        statistics.quantiles(latencies_ms, n=100)[94],
        max(latencies_ms),
    )

    if cpu_memory is not None:
        logging.info("CPU memory (RSS): %.2f MiB", cpu_memory)
    else:  # pragma: no cover - psutil missing
        logging.info("CPU memory measurement skipped (psutil not installed)")

    if gpu_memory is not None:
        logging.info("GPU memory (peak allocated): %.2f MiB", gpu_memory)
    else:
        logging.info("GPU memory measurement unavailable for device=%s", device)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
