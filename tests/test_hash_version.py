from __future__ import annotations

import importlib
from pathlib import Path

import pytest

qdrant_client_module = pytest.importorskip("qdrant_client")
QdrantClient = qdrant_client_module.QdrantClient


def _point_count(client: QdrantClient, collection: str) -> int:
    return client.count(collection_name=collection, exact=True).count


def test_duplicate_upload_reuses_vectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    collection_name = "hash_version_test"
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()

    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", str(qdrant_path))
    monkeypatch.setenv("QDRANT_COLLECTION", collection_name)
    monkeypatch.setenv("EMBED_MODEL", "mock")

    import core.config as config

    importlib.reload(config)
    from core.index import index_path

    text = tmp_path / "book.txt"
    text.write_text("lorem ipsum dolor sit amet", encoding="utf-8")

    first = index_path(text)
    assert first.new > 0
    assert first.indexed_chunks >= first.new

    qdrant_client = QdrantClient(path=str(qdrant_path))
    initial_count = _point_count(qdrant_client, collection_name)
    assert initial_count == first.indexed_chunks

    second = index_path(text)
    assert second.new == 0
    assert second.skipped == first.indexed_chunks
    assert second.book_id == first.book_id
    assert second.version == first.version
    assert second.file_hash == first.file_hash

    final_count = _point_count(qdrant_client, collection_name)
    assert final_count == initial_count
