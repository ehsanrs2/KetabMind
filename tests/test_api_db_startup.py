from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.helpers import setup_api_stubs


def _reload_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", str(tmp_path / "qdrant"))
    monkeypatch.setenv("QDRANT_COLLECTION", "startup-tests")
    monkeypatch.setenv("EMBED_MODEL", "mock")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))

    setup_api_stubs(monkeypatch)

    import core.config as config

    importlib.reload(config)

    to_remove = [name for name in list(sys.modules) if name.startswith("apps.api.")]
    for name in to_remove:
        sys.modules.pop(name, None)

    api_main = importlib.import_module("apps.api.main")
    return importlib.reload(api_main)


def test_startup_initializes_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api_main = _reload_api(tmp_path, monkeypatch)
    app = api_main.app  # noqa: WPS433

    with TestClient(app) as client:
        bookmarks_response = client.get("/bookmarks")
        assert bookmarks_response.status_code == 200
        assert "bookmarks" in bookmarks_response.json()

        sessions_response = client.get("/sessions")
        assert sessions_response.status_code == 200
        assert "sessions" in sessions_response.json()

        upload_response = client.post(
            "/upload",
            files={"file": ("tiny.txt", b"hello world", "text/plain")},
        )
        assert upload_response.status_code == 200

        books_response = client.get("/books")
        assert books_response.status_code == 200
        assert "books" in books_response.json()
