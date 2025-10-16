"""GPU optimization benchmark script."""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import time
from dataclasses import dataclass

import torch

from embedding.adapter import EmbeddingAdapter
from utils.gpu_opt import GPUOptimizer, quantization_config


@dataclass
class BenchmarkResult:
    batch_size: int
    latencies_ms: list[float]
    gpu_memory_bytes: int | None

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies_ms)

    @property
    def p50_latency(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def p95_latency(self) -> float:
        if len(self.latencies_ms) == 1:
            return self.latencies_ms[0]
        return statistics.quantiles(self.latencies_ms, n=100)[94]

    @property
    def max_latency(self) -> float:
        return max(self.latencies_ms)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s: %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GPU optimization pipeline")
    parser.add_argument("--model-name", default=os.getenv("EMBED_MODEL_NAME", "bge-m3"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--sample-text", default="KetabMind GPU benchmark sample")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _prepare_adapter(model_name: str, batch_size: int, device: str | None) -> EmbeddingAdapter:
    os.environ["EMBED_MODEL_NAME"] = model_name
    os.environ["BATCH_SIZE"] = str(batch_size)
    return EmbeddingAdapter(model_name=model_name, batch_size=batch_size, device=device)


def _run_benchmark(
    adapter: EmbeddingAdapter,
    texts: list[str],
    repeats: int,
    warmup: int,
) -> BenchmarkResult:
    if torch.cuda.is_available() and "cuda" in adapter.device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        adapter.embed_texts(texts, batch_size=len(texts))
        if torch.cuda.is_available() and "cuda" in adapter.device:
            torch.cuda.synchronize()

    latencies: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        adapter.embed_texts(texts, batch_size=len(texts))
        if torch.cuda.is_available() and "cuda" in adapter.device:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    gpu_memory = None
    if torch.cuda.is_available() and "cuda" in adapter.device:
        gpu_memory = torch.cuda.max_memory_allocated()

    return BenchmarkResult(
        batch_size=len(texts),
        latencies_ms=latencies,
        gpu_memory_bytes=gpu_memory,
    )


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    if args.quantization != "none":
        os.environ["EMBED_QUANT"] = args.quantization
        try:
            quant_cfg = quantization_config(args.quantization)
            logging.info("Using quantization: mode=%s config=%s", args.quantization, quant_cfg)
        except ImportError as exc:
            logging.warning("Quantization requested but bitsandbytes missing: %s", exc)
    elif "EMBED_QUANT" in os.environ:
        del os.environ["EMBED_QUANT"]

    base_adapter = _prepare_adapter(args.model_name, args.batch_size, args.device)
    texts = [args.sample_text for _ in range(args.batch_size)]

    logging.info("Running baseline benchmark (batch=%d)", args.batch_size)
    baseline = _run_benchmark(base_adapter, texts, args.repeats, args.warmup)

    optimizer = GPUOptimizer(device=base_adapter.device)
    available_memory = optimizer.get_available_memory()
    logging.info(
        "Detected available GPU memory: %.2f MiB",
        available_memory / (1024**2) if available_memory else 0.0,
    )

    optimized_batch = args.batch_size
    if baseline.gpu_memory_bytes and baseline.gpu_memory_bytes > 0:
        per_sample = baseline.gpu_memory_bytes / max(args.batch_size, 1)
        optimized_batch = optimizer.adjust_batch_size(args.batch_size, int(per_sample))
        if optimized_batch != args.batch_size:
            logging.info("Adjusted batch size from %d to %d", args.batch_size, optimized_batch)
    else:
        logging.info("Skipping batch adjustment; GPU stats unavailable.")

    optimized_result = baseline
    if optimized_batch != args.batch_size:
        optimized_adapter = _prepare_adapter(args.model_name, optimized_batch, args.device)
        opt_texts = [args.sample_text for _ in range(optimized_batch)]
        logging.info("Running optimized benchmark (batch=%d)", optimized_batch)
        optimized_result = _run_benchmark(optimized_adapter, opt_texts, args.repeats, args.warmup)

    def _log_result(label: str, result: BenchmarkResult) -> None:
        logging.info(
            "%s -> latency avg: %.2f ms | p50: %.2f ms | p95: %.2f ms | max: %.2f ms",
            label,
            result.avg_latency,
            result.p50_latency,
            result.p95_latency,
            result.max_latency,
        )
        if result.gpu_memory_bytes:
            logging.info(
                "%s -> peak GPU memory: %.2f MiB",
                label,
                result.gpu_memory_bytes / (1024**2),
            )
        else:
            logging.info("%s -> GPU memory stats unavailable", label)

    _log_result("Baseline", baseline)
    if optimized_result is not baseline:
        _log_result("Optimized", optimized_result)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
