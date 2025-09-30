"""Benchmark retrieval latency with and without reranker cache."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.retrieve.retriever import Retriever

QUERY_COUNT = 100
CANDIDATE_COUNT = 50


class MockEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]


@dataclass
class MockHit:
    payload: dict[str, object]
    score: float
    id: str


class MockClient:
    def search(self, collection_name: str, query_vector: Sequence[float], limit: int) -> list[MockHit]:
        return [
            MockHit(
                payload={
                    "text": f"Candidate {idx} for {collection_name}",
                    "book_id": "book",
                    "page_num": idx,
                    "chunk_id": f"chunk-{idx}",
                },
                score=1.0 - (idx * 0.001),
                id=f"chunk-{idx}",
            )
            for idx in range(limit)
        ]


class MockStore:
    def __init__(self) -> None:
        self.client = MockClient()
        self.collection = "mock"

    def ensure_collection(self, dim: int) -> None:
        return None


class MockReranker:
    def __init__(self, delay: float = 0.002) -> None:
        self.delay = delay
        self.timeout_s = 5.0

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        time.sleep(self.delay)
        return [0.5 for _ in pairs]


def _run_queries(retriever: Retriever, queries: Sequence[str]) -> float:
    start = time.perf_counter()
    for query in queries:
        retriever.retrieve(query)
    return time.perf_counter() - start


def main() -> None:
    queries = [f"query {idx % 10}" for idx in range(QUERY_COUNT)]

    embedder = MockEmbedder()

    no_cache_retriever = Retriever(
        top_m=CANDIDATE_COUNT,
        reranker_topk=CANDIDATE_COUNT,
        embedder=embedder,
        store=MockStore(),
        reranker=MockReranker(),
        reranker_enabled=True,
        reranker_cache_size=0,
    )
    without_cache = _run_queries(no_cache_retriever, queries)

    cache_retriever = Retriever(
        top_m=CANDIDATE_COUNT,
        reranker_topk=CANDIDATE_COUNT,
        embedder=embedder,
        store=MockStore(),
        reranker=MockReranker(),
        reranker_enabled=True,
        reranker_cache_size=1024,
    )
    _run_queries(cache_retriever, queries)  # warm cache
    with_cache = _run_queries(cache_retriever, queries)

    print(f"Without cache: {without_cache:.3f}s for {QUERY_COUNT} queries")
    print(f"With cache:    {with_cache:.3f}s for {QUERY_COUNT} queries")


if __name__ == "__main__":
    main()
