from __future__ import annotations

import importlib
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_upload_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", str(tmp_path))
    monkeypatch.setenv("QDRANT_COLLECTION", "tapiu")
    monkeypatch.setenv("EMBED_MODEL", "mock")

    import core.config as config

    importlib.reload(config)
    from apps.api.main import app  # noqa: WPS433

    client = TestClient(app)

    def files() -> dict[str, tuple[str, io.BytesIO, str]]:
        return {"file": ("tiny.txt", io.BytesIO(b"hello world"), "text/plain")}

    r1 = client.post("/upload", files=files())
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["new"] > 0

    r2 = client.post("/upload", files=files())
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["skipped"] > 0
