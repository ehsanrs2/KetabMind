from __future__ import annotations

import importlib
from pathlib import Path

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

qdrant_client_module = pytest.importorskip("qdrant_client")
QdrantClient = qdrant_client_module.QdrantClient


def _collection_count(client: QdrantClient, collection: str) -> int:
    return client.count(collection_name=collection, exact=True).count


def test_duplicate_pdf_upload_does_not_increase_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection_name = "duplicate_pdf_test"
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()

    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", str(qdrant_path))
    monkeypatch.setenv("QDRANT_COLLECTION", collection_name)
    monkeypatch.setenv("EMBED_MODEL", "mock")

    import core.config as config

    importlib.reload(config)
    from apps.api.main import app  # noqa: WPS433

    client = TestClient(app)

    src_pdf = Path("docs/fixtures/sample.pdf")
    pdf_path = tmp_path / src_pdf.name
    pdf_path.write_bytes(src_pdf.read_bytes())

    payload = {"path": str(pdf_path)}

    first_response = client.post("/index", json=payload)
    assert first_response.status_code == 200
    first_data = first_response.json()
    assert first_data["new"] > 0
    assert first_data["book_id"]
    assert first_data["version"].startswith("v")

    qdrant_client = QdrantClient(path=str(qdrant_path))
    initial_count = _collection_count(qdrant_client, collection_name)
    assert initial_count > 0

    second_response = client.post("/index", json=payload)
    assert second_response.status_code == 200
    second_data = second_response.json()
    assert second_data["skipped"] > 0
    assert second_data["book_id"] == first_data["book_id"]
    assert second_data["version"] == first_data["version"]

    final_count = _collection_count(qdrant_client, collection_name)
    assert final_count == initial_count
