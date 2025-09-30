from __future__ import annotations

import importlib
from pathlib import Path

import pytest

def _make_pdf(path: Path, *, text: str) -> None:
    from reportlab.lib.pagesizes import letter  # type: ignore import-not-found
    from reportlab.pdfgen import canvas  # type: ignore import-not-found

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(100, 750, text)
    c.showPage()
    c.save()


def test_index_query_phase1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("qdrant_client")
    pytest.importorskip("reportlab")

    from fastapi.testclient import TestClient  # type: ignore import-not-found
    from qdrant_client import QdrantClient  # type: ignore import-not-found

    collection = "e2e-phase1"
    store_path = tmp_path / "qdrant"

    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", str(store_path))
    monkeypatch.setenv("QDRANT_COLLECTION", collection)
    monkeypatch.setenv("EMBED_MODEL", "mock")
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setenv("LLM_MODEL", "mock")

    import core.config as config

    importlib.reload(config)
    config.reload_settings()

    # Reload modules that depend on configuration to ensure clean state.
    import core.index.indexer as indexer
    import core.vector.qdrant_client as ingest_store
    import core.vector.qdrant as query_store
    import core.answer.answerer as answerer
    import apps.api.main as api_main

    importlib.reload(indexer)
    importlib.reload(ingest_store)
    importlib.reload(query_store)
    importlib.reload(answerer)
    importlib.reload(api_main)

    client = TestClient(api_main.app)

    pdf_path = tmp_path / "books" / "phase1.pdf"
    pdf_text = "KetabMind integration phase one reference text"
    _make_pdf(pdf_path, text=pdf_text)

    metadata = {"author": "Phase One", "year": "2024", "subject": "Integration"}

    with pdf_path.open("rb") as handle:
        upload_response = client.post(
            "/upload",
            files={"file": (pdf_path.name, handle, "application/pdf")},
            data=metadata,
        )

    assert upload_response.status_code == 200
    upload_payload = upload_response.json()
    book_id = upload_payload["book_id"]
    version = upload_payload["version"]
    upload_path = Path(upload_payload["path"])

    index_response = client.post(
        "/index",
        json={"path": str(upload_path), **metadata},
    )

    assert index_response.status_code == 200
    index_payload = index_response.json()
    assert index_payload["book_id"] == book_id
    assert index_payload["version"] == version
    assert index_payload["file_hash"] == upload_payload["file_hash"]

    q_client = QdrantClient(path=str(store_path))
    try:
        points, _ = q_client.scroll(collection_name=collection, limit=20)
    finally:
        close = getattr(q_client, "close", None)
        if callable(close):
            close()
    payloads = [pt.payload or {} for pt in points]
    assert payloads, "No payloads returned from Qdrant"

    target_payload = next((p for p in payloads if p.get("book_id") == book_id), None)
    assert target_payload is not None, "Indexed payload with expected book_id not found"
    assert target_payload.get("version") == version
    assert target_payload.get("page_num") is not None
    meta = target_payload.get("meta") or {}
    assert meta.get("author") == metadata["author"]
    assert meta.get("subject") == metadata["subject"]

    query_response = client.post(
        "/query",
        json={"q": "phase one reference text", "top_k": 1},
    )

    assert query_response.status_code == 200
    query_payload = query_response.json()
    contexts = query_payload.get("contexts", [])
    assert contexts, "Query response did not include any contexts"
    citation = contexts[0]
    assert citation["book_id"] == book_id
    assert citation["version"] == version
    assert citation.get("page_num") == target_payload.get("page_num")
