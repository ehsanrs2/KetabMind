from typing import Any, List

import pytest

from core.vector import qdrant


class DummyClient:
    def __init__(self) -> None:
        self.upserts: list[Any] = []

    def upsert(self, collection_name: str, points: List[Any]) -> None:
        self.upserts.extend(points)

    def search(
        self, collection_name: str, query_vector: List[float], limit: int
    ) -> List[Any]:
        class Hit:
            def __init__(self, payload: dict[str, Any]) -> None:
                self.payload = payload

        return [Hit({"text": "dummy"})]


def test_vector_store(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyClient()
    monkeypatch.setattr(qdrant, "QdrantClient", lambda url: dummy)
    store = qdrant.VectorStore()
    store.upsert([[0.0, 0.0, 0.0]], [{"text": "a"}])
    assert dummy.upserts
    res = store.query([0.0, 0.0, 0.0], 1)
    assert res[0]["text"] == "dummy"
