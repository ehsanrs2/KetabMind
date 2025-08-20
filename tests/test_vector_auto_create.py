import pytest
from typing import cast

import core.vector.qdrant as qdrant


def test_upsert_creates_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qdrant.settings, "qdrant_mode", "local")  # type: ignore[attr-defined]
    monkeypatch.setattr(qdrant.settings, "qdrant_url", None)  # type: ignore[attr-defined]
    store = qdrant.VectorStore()
    try:
        store.client.delete_collection(collection_name=store.collection)
    except Exception:
        pass
    with pytest.raises(Exception):
        store.client.get_collection(store.collection)
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
    new, _ = store.upsert([[0.0, 0.0, 0.0]], [payload])
    assert new > 0
    store.client.get_collection(store.collection)
