from __future__ import annotations

import os
import tempfile

from fastapi.testclient import TestClient


def test_health() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["QDRANT_MODE"] = "local"
        os.environ["QDRANT_LOCATION"] = tmp
        os.environ["QDRANT_COLLECTION"] = "tapi"
        os.environ["EMBED_MODEL"] = "small"
        # import after env set
        from apps.api.main import app  # noqa: WPS433

        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
