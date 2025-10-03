from __future__ import annotations

import importlib
import io
from pathlib import Path

import pytest

from fastapi.testclient import TestClient
from tests.helpers import setup_api_stubs


def test_upload_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", str(tmp_path))
    monkeypatch.setenv("QDRANT_COLLECTION", "tapiu")
    monkeypatch.setenv("EMBED_MODEL", "mock")
    upload_dir = tmp_path / "uploads"
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))

    setup_api_stubs(monkeypatch)

    import core.config as config

    importlib.reload(config)
    import apps.api.main as api_main

    importlib.reload(api_main)
    app = api_main.app  # noqa: WPS433

    client = TestClient(app)

    def files() -> dict[str, tuple[str, io.BytesIO, str]]:
        return {"file": ("tiny.txt", io.BytesIO(b"hello world"), "text/plain")}

    r1 = client.post("/upload", files=files())
    assert r1.status_code == 200
    data1 = r1.json()
    assert set(data1.keys()) == {"book_id", "version", "file_hash"}
    assert data1["book_id"]
    assert data1["file_hash"].startswith("sha256:")
    assert data1["version"].startswith("v")

    r2 = client.post("/upload", files=files())
    assert r2.status_code == 200
    data2 = r2.json()
    assert set(data2.keys()) == {"book_id", "version", "file_hash"}
    assert data2["book_id"] == data1["book_id"]
    assert data2["version"] == data1["version"]
