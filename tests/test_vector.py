from typing import Any, List, cast

import pytest
from qdrant_client.http import models as rest

import core.vector.qdrant as qdrant


class DummyClient:
    def __init__(self) -> None:
        self.upserts: list[Any] = []
        self.collection_exists = False

    def get_collection(self, collection_name: str) -> None:
        if not self.collection_exists:
            raise Exception("missing")

    def recreate_collection(self, collection_name: str, vectors_config: Any) -> None:
        self.collection_exists = True

    def upsert(self, collection_name: str, points: List[Any]) -> None:
        self.upserts.extend(points)

    def scroll(
        self, collection_name: str, scroll_filter: Any, limit: int, **_: Any
    ) -> tuple[List[Any], None]:
        return ([], None)

    def search(
        self, collection_name: str, query_vector: List[float], limit: int
    ) -> List[Any]:
        class Hit:
            def __init__(self, payload: dict[str, Any]) -> None:
                self.payload = payload

        return [Hit({"text": "dummy"})]


def test_vector_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qdrant.settings, "qdrant_mode", "local")  # type: ignore[attr-defined]
    monkeypatch.setattr(qdrant.settings, "qdrant_location", ":memory:")  # type: ignore[attr-defined]
    dummy = DummyClient()
    monkeypatch.setattr(qdrant, "QdrantClient", lambda *a, **k: dummy)
    store = qdrant.VectorStore()
    payload = cast(
        qdrant.ChunkPayload,
        {
            "text": "a",
            "book_id": "b",
            "chapter": None,
            "page_start": 1,
            "page_end": 1,
            "chunk_id": "c1",
            "content_hash": "h",
        },
    )
    new, skipped = store.upsert([[0.0, 0.0, 0.0]], [payload])
    assert new == 1 and skipped == 0 and dummy.upserts
    res = store.query([0.0, 0.0, 0.0], 1)
    assert res[0]["text"] == "dummy"


def test_vector_store_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qdrant.settings, "qdrant_mode", "local")  # type: ignore[attr-defined]
    monkeypatch.setattr(qdrant.settings, "qdrant_url", None)  # type: ignore[attr-defined]
    monkeypatch.setattr(qdrant.settings, "qdrant_location", ":memory:")  # type: ignore[attr-defined]
    store = qdrant.VectorStore()
    store.client.recreate_collection(
        collection_name=store.collection,
        vectors_config=rest.VectorParams(size=3, distance=rest.Distance.COSINE),
    )
    payload = cast(
        qdrant.ChunkPayload,
        {
            "text": "test",
            "book_id": "b",
            "chapter": None,
            "page_start": 1,
            "page_end": 1,
            "chunk_id": "c1",
            "content_hash": "h",
        },
    )
    new, skipped = store.upsert([[0.0, 0.0, 0.0]], [payload])
    assert new == 1 and skipped == 0
    res = store.query([0.0, 0.0, 0.0], 1)
    assert res[0]["text"] == "test"
