from __future__ import annotations

import importlib
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.helpers import setup_api_stubs


def _make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ttl: int | None = None) -> TestClient:
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", str(tmp_path / "qdrant"))
    monkeypatch.setenv("QDRANT_COLLECTION", "viewer")
    monkeypatch.setenv("EMBED_MODEL", "mock")
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    if ttl is not None:
        monkeypatch.setenv("UPLOAD_SIGNED_URL_TTL", str(ttl))

    setup_api_stubs(monkeypatch)

    import core.config as config
    import apps.api.main as api_main

    importlib.reload(config)
    importlib.reload(api_main)
    return TestClient(api_main.app)


def _upload_sample(client: TestClient) -> dict[str, str]:
    files = {"file": ("tiny.txt", io.BytesIO(b"hello world"), "text/plain")}
    response = client.post("/upload", files=files, data={"title": "Tiny"})
    assert response.status_code == 200
    return response.json()


def test_signed_url_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(tmp_path, monkeypatch)
    data = _upload_sample(client)
    book_id = data["book_id"]

    stored_path = tmp_path / "uploads" / "user-alice" / book_id / "tiny.txt"
    assert stored_path.read_bytes() == b"hello world"

    view_resp = client.get(f"/book/{book_id}/page/1/view")
    assert view_resp.status_code == 200
    payload = view_resp.json()
    assert payload["expires_at"] > 0
    signed_url = payload["url"]

    fetch_resp = client.get(signed_url)
    assert fetch_resp.status_code == 200
    assert fetch_resp.content == b"hello world"
    assert fetch_resp.headers["content-type"].startswith("text/plain")


def test_signed_url_expiry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(tmp_path, monkeypatch, ttl=1)
    data = _upload_sample(client)
    book_id = data["book_id"]

    view_resp = client.get(f"/book/{book_id}/page/1/view")
    assert view_resp.status_code == 200
    signed_url = view_resp.json()["url"]

    import apps.api.main as api_main

    current = api_main.time.time()
    monkeypatch.setattr(api_main.time, "time", lambda: current + 10)

    expired_resp = client.get(signed_url)
    assert expired_resp.status_code == 403


def test_view_acl_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client(tmp_path, monkeypatch)
    data = _upload_sample(client)
    book_id = data["book_id"]

    import apps.api.auth.jwt as jwt

    monkeypatch.setattr(jwt, "_DEFAULT_USER_ID", "user-bob")

    forbidden = client.get(f"/book/{book_id}/page/1/view")
    assert forbidden.status_code == 404
