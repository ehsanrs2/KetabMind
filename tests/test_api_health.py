from __future__ import annotations

from apps.api.main import app
from fastapi.testclient import TestClient


def test_health_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
