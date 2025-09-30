from __future__ import annotations

from dataclasses import dataclass

from core.retrieve.retriever import Retriever


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * 3 for _ in texts]


@dataclass
class FakeHit:
    id: str
    score: float
    text: str
    book_id: str
    page: int

    @property
    def payload(self) -> dict[str, object]:
        return {
            "text": self.text,
            "book_id": self.book_id,
            "page_start": self.page,
            "chunk_id": self.id,
            "meta": {},
        }


class FakeClient:
    def __init__(self, hits: list[FakeHit]) -> None:
        self._hits = hits

    def search(self, collection_name: str, query_vector: list[float], limit: int) -> list[FakeHit]:
        return list(self._hits[:limit])


class FakeStore:
    def __init__(self, hits: list[FakeHit]) -> None:
        self.client = FakeClient(hits)
        self.collection = "books"
        self.dim: int | None = None

    def ensure_collection(self, dim: int) -> None:
        self.dim = dim


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(list(pairs))
        return list(self._scores)


def _hits() -> list[FakeHit]:
    return [
        FakeHit("c1", 0.6, "alpha beta", "b1", 1),
        FakeHit("c2", 0.9, "alpha gamma", "b2", 2),
        FakeHit("c3", 0.7, "delta", "b3", 3),
    ]


def test_hybrid_weights_change_order() -> None:
    query = "alpha beta"
    hits = _hits()
    embedder = FakeEmbedder()
    reranker = FakeReranker([0.4, 0.8])

    retriever = Retriever(
        top_n=3,
        top_m=3,
        reranker_topk=2,
        embedder=embedder,
        store=FakeStore(hits),
        reranker=reranker,
        hybrid_weights={"cosine": 0.4, "lexical": 0.2, "reranker": 0.4},
        reranker_enabled=True,
    )

    results_default = retriever.retrieve(query)
    assert [chunk.id for chunk in results_default] == ["c2", "c1", "c3"]
    assert reranker.calls
    assert len(reranker.calls[-1]) == 2

    lexical_reranker = FakeReranker([0.0, 0.0])
    lexical_first = Retriever(
        top_n=3,
        top_m=3,
        reranker_topk=2,
        embedder=embedder,
        store=FakeStore(hits),
        reranker=lexical_reranker,
        hybrid_weights={"cosine": 0.0, "lexical": 1.0, "reranker": 0.0},
        reranker_enabled=False,
    )

    results_lexical = lexical_first.retrieve(query)
    assert [chunk.id for chunk in results_lexical] == ["c1", "c2", "c3"]
