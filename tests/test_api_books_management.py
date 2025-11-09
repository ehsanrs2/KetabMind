import importlib
import io
from pathlib import Path
from typing import Any

import pytest

from fastapi.testclient import TestClient

from core.vector.qdrant_client import VectorStore
from qdrant_client.http import models as rest
from tests.helpers import setup_api_stubs


def _scroll_points(store: VectorStore, book_id: str) -> list[dict[str, Any]]:
    filter_payload = rest.Filter(
        must=[rest.FieldCondition(key="book_id", match=rest.MatchValue(value=book_id))]
    )
    kwargs: dict[str, Any] = {
        "collection_name": store.collection,
        "scroll_filter": filter_payload,
        "limit": 128,
        "with_payload": True,
        "with_vectors": False,
    }
    try:
        points, _ = store.client.scroll(**kwargs)
    except TypeError:
        kwargs.pop("scroll_filter", None)
        kwargs["filter"] = filter_payload
        points, _ = store.client.scroll(**kwargs)
    return [dict(payload=p.payload or {}) for p in points]


def test_book_management_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_MODE", "local")
    qdrant_dir = tmp_path / "qdrant"
    monkeypatch.setenv("QDRANT_LOCATION", str(qdrant_dir))
    monkeypatch.setenv("QDRANT_COLLECTION", "books-test")
    monkeypatch.setenv("EMBED_MODEL", "mock")
    upload_dir = tmp_path / "uploads"
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'app.db'}")

    setup_api_stubs(monkeypatch)

    import core.config as config

    importlib.reload(config)
    import apps.api.main as api_main

    importlib.reload(api_main)

    from apps.api.db.base import Base
    from apps.api.db.session import _engine

    Base.metadata.create_all(_engine)

    app = api_main.app  # noqa: WPS433

    client = TestClient(app)

    file_payload = {"file": ("tiny.txt", io.BytesIO(b"hello world"), "text/plain")}
    upload_response = client.post("/upload", files=file_payload)
    assert upload_response.status_code == 200
    upload_data = upload_response.json()
    book_id = upload_data["book_id"]

    list_response = client.get("/books")
    assert list_response.status_code == 200
    books_payload = list_response.json()
    assert books_payload["total"] == 1
    summary = books_payload["books"][0]
    assert summary["id"] == book_id
    assert summary["status"] == "indexed"

    rename_payload = {"title": "My Updated Book"}
    rename_response = client.patch(f"/books/{book_id}/rename", json=rename_payload)
    assert rename_response.status_code == 200
    rename_data = rename_response.json()["book"]
    assert rename_data["title"] == "My Updated Book"

    detail_response = client.get(f"/books/{book_id}")
    assert detail_response.status_code == 200
    detail_data = detail_response.json()["book"]
    assert detail_data["title"] == "My Updated Book"

    with VectorStore(
        mode="local",
        location=str(qdrant_dir),
        url=None,
        collection="books-test",
        ensure_collection=False,
        vector_size=None,
    ) as store:
        points = _scroll_points(store, book_id)
        assert points, "Expected book vectors to exist"
        if not hasattr(store.client, "set_payload"):
            pytest.skip("Vector client does not support in-place payload updates")
        for point in points:
            meta = point.get("meta") or {}
            assert meta.get("title") == "My Updated Book"

    delete_response = client.delete(f"/books/{book_id}")
    assert delete_response.status_code == 204

    missing_response = client.get(f"/books/{book_id}")
    assert missing_response.status_code == 404

    with VectorStore(
        mode="local",
        location=str(qdrant_dir),
        url=None,
        collection="books-test",
        ensure_collection=False,
        vector_size=None,
    ) as store:
        remaining = _scroll_points(store, book_id)
        assert not remaining
