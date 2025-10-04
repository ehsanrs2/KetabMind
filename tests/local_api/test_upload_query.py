"""Integration test for the lightweight local FastAPI application."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def _load_backend():
    """Load (or reload) the backend module for testing."""
    if "backend.main" in sys.modules:
        del sys.modules["backend.main"]

    module = importlib.import_module("backend.main")
    importlib.reload(module)
    return module


def test_upload_index_query(tmp_path, monkeypatch):
    monkeypatch.setenv("KETABMIND_BOOK_DIR", str(tmp_path))
    backend_main = _load_backend()

    class DummyLLM:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return f"dummy-response:{prompt}"

    dummy_llm = DummyLLM()
    monkeypatch.setattr(backend_main, "local_llm", dummy_llm, raising=False)

    client = TestClient(backend_main.app)

    pdf_bytes = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sample.pdf"
    stored_path = Path(data["path"])
    assert stored_path.exists()
    assert stored_path.read_bytes() == pdf_bytes

    index_response = client.post("/index", json={"filename": "sample.pdf"})
    assert index_response.status_code == 200
    assert index_response.json() == {"filename": "sample.pdf", "indexed": True}

    query_response = client.post("/query", json={"prompt": "Tell me something"})
    assert query_response.status_code == 200
    assert query_response.json() == {"response": "dummy-response:Tell me something"}
    assert dummy_llm.prompts == ["Tell me something"]

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    version_response = client.get("/version")
    assert version_response.status_code == 200
    assert "version" in version_response.json()
