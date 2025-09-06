from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_cors_headers() -> None:
    import core.config as config

    importlib.reload(config)
    from apps.api.main import app  # noqa: WPS433

    client = TestClient(app)
    headers = {
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "POST",
    }
    r = client.options("/upload", headers=headers)
    assert r.status_code in (200, 204)
    assert r.headers.get("access-control-allow-origin") == "http://example.com"
