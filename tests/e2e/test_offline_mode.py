from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
from requests import RequestException


def _install_stub_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install lightweight stubs for optional heavy dependencies when absent."""

    if importlib.util.find_spec("torch") is None and "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

    if importlib.util.find_spec("transformers") is None and "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")

        class _DummyTokenizer:
            @classmethod
            def from_pretrained(cls, *args: object, **kwargs: object) -> "_DummyTokenizer":
                return cls()

        class _DummyModel:
            @classmethod
            def from_pretrained(cls, *args: object, **kwargs: object) -> "_DummyModel":
                return cls()

        def _pipeline(*args: object, **kwargs: object):
            def _generate(prompt: str, *_, **__) -> list[dict[str, str]]:
                return [{"generated_text": f"{prompt} offline"}]

            return _generate

        transformers_stub.AutoTokenizer = _DummyTokenizer
        transformers_stub.AutoModelForCausalLM = _DummyModel
        transformers_stub.pipeline = _pipeline
        monkeypatch.setitem(sys.modules, "transformers", transformers_stub)


def test_offline_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("pydantic")

    _install_stub_modules(monkeypatch)

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("KETABMIND_BOOK_DIR", str(home_dir / ".ketabmind" / "books"))

    from fastapi.testclient import TestClient  # type: ignore import-not-found

    import backend.local_llm as local_llm
    import backend.main as api_main

    log_path = Path.home() / ".ketabmind" / "logs" / "app.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8")],
        force=True,
    )

    def _offline_post(*args: object, **kwargs: object):
        raise RequestException("network disabled for offline mode test")

    monkeypatch.setattr(local_llm.requests, "post", _offline_post)

    monkeypatch.setattr(
        local_llm,
        "_hf_generate",
        lambda prompt, model_name=None: "Offline fallback response [book:1-1]",
    )

    client = TestClient(api_main.app)

    book_bytes = b"Offline test content"
    upload_response = client.post(
        "/upload",
        files={"file": ("offline.txt", book_bytes, "text/plain")},
    )
    assert upload_response.status_code == 200
    upload_payload = upload_response.json()
    stored_path = Path(upload_payload["path"])
    assert stored_path.read_bytes() == book_bytes

    index_response = client.post(
        "/index",
        json={"filename": upload_payload["filename"]},
    )
    assert index_response.status_code == 200
    assert index_response.json()["indexed"] is True

    query_response = client.post(
        "/query",
        json={"prompt": "Summarise offline capabilities"},
    )
    assert query_response.status_code == 200
    assert "Offline fallback response" in query_response.json()["response"]

    logging.shutdown()
    log_contents = log_path.read_text(encoding="utf-8")
    assert "Ollama request failed" in log_contents
    assert "Falling back to HuggingFace" in log_contents
