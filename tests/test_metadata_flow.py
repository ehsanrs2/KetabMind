from __future__ import annotations

import importlib
import io
import json
from pathlib import Path

import pytest

from core.vector.qdrant_client import VectorStore
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_metadata_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", str(tmp_path))
    monkeypatch.setenv("QDRANT_COLLECTION", "meta_flow")
    monkeypatch.setenv("EMBED_MODEL", "mock")

    import core.config as config

    importlib.reload(config)
    config.reload_settings()
    from apps.api.main import app  # noqa: WPS433
    from core.config import settings

    client = TestClient(app)

    def files() -> dict[str, tuple[str, io.BytesIO, str]]:
        return {"file": ("tiny.txt", io.BytesIO(b"hello world"), "text/plain")}

    metadata = {"author": "Jane Austen", "year": "1813", "subject": "Fiction"}

    response = client.post("/upload", files=files(), data=metadata)
    assert response.status_code == 200
    payload = response.json()

    book_id = payload["book_id"]
    jsonl_path = Path(settings.qdrant_location) / f"{book_id}.jsonl"
    assert jsonl_path.exists()

    lines = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines, "expected at least one JSONL record"
    for record in lines:
        assert record["book_id"] == book_id
        assert record["version"] == payload["version"]
        assert record["file_hash"] == payload["file_hash"]
        assert record["meta"] == metadata

    with VectorStore(
        mode=settings.qdrant_mode,
        location=settings.qdrant_location,
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        vector_size=1,
    ) as store:
        points, _ = store.client.scroll(
            collection_name=settings.qdrant_collection,
            limit=10,
        )

    metas = [dict(point.payload or {}).get("meta") for point in points]
    assert metas, "expected metadata in Qdrant payloads"
    assert any(meta == metadata for meta in metas)
