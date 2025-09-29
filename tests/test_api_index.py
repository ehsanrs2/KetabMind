from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_index_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", str(tmp_path))
    monkeypatch.setenv("QDRANT_COLLECTION", "tapi")
    monkeypatch.setenv("EMBED_MODEL", "mock")

    import core.config as config

    importlib.reload(config)
    from apps.api.main import app  # noqa: WPS433

    client = TestClient(app)
    txt = tmp_path / "tiny.txt"
    txt.write_text("hello world", encoding="utf-8")

    payload = {"path": str(txt)}
    r1 = client.post("/index", json=payload)
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["new"] > 0
    assert data1["book_id"]
    assert data1["version"].startswith("v")
    assert data1["file_hash"].startswith("sha256:")
    assert data1["indexed_chunks"] >= data1["new"]

    r2 = client.post("/index", json=payload)
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["skipped"] > 0
    assert data2["book_id"] == data1["book_id"]
    assert data2["version"] == data1["version"]
