from __future__ import annotations

from collections.abc import Sequence

import pytest

from core.retrieve.retriever import Retriever, SearchClient, VectorStoreLike


def test_retriever_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0]]

    monkeypatch.setattr("core.retrieve.retriever.get_embedder", lambda: DummyEmbedder())

    class Hit:
        def __init__(self, payload: dict[str, str | int], score: float) -> None:
            self.payload = payload
            self.score = score

    class DummyClient(SearchClient):
        def search(
            self, collection_name: str, query_vector: Sequence[float], limit: int
        ) -> list[Hit]:
            return [
                Hit(
                    {
                        "text": "foo qux",
                        "book_id": "b1",
                        "page_start": 0,
                        "page_end": 0,
                    },
                    0.19,
                ),
                Hit(
                    {
                        "text": "foo bar",
                        "book_id": "b2",
                        "page_start": 0,
                        "page_end": 0,
                    },
                    0.20,
                ),
            ]

    class DummyStore:
        def __init__(self) -> None:
            self.client: SearchClient = DummyClient()
            self.collection = "c"

    store: VectorStoreLike = DummyStore()
    r = Retriever(top_k=2)
    r.store = store
    res = r.retrieve("foo bar", top_k=2)
    assert res[0].text == "foo bar"
    assert res[0].score > res[1].score
    assert res[0].distance == pytest.approx(0.80)
