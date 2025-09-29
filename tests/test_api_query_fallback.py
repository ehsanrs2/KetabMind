from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from core.answer import answerer


def _noop_index_path(*args: object, **kwargs: object) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        new=0,
        skipped=0,
        collection="stub",
        book_id="stub",
        version="v0",
        file_hash="sha256:stub",
        indexed_chunks=0,
    )


def _noop_find_indexed_file(*args: object, **kwargs: object) -> None:
    return None


def _ensure_core_index_stub() -> None:
    if "core.index" in sys.modules:
        return
    fake_index_module = types.ModuleType("core.index")
    fake_index_module.index_path = _noop_index_path  # type: ignore[attr-defined]
    fake_index_module.IndexResult = types.SimpleNamespace  # type: ignore[attr-defined]
    fake_index_module.find_indexed_file = _noop_find_indexed_file  # type: ignore[attr-defined]
    sys.modules["core.index"] = fake_index_module


class EmptyRetriever:
    def retrieve(self, query: str, top_k: int) -> list[object]:
        return []


class FallbackLLM:
    def generate(self, prompt: str, stream: bool = False) -> str:
        return "Not enough information to answer accurately."


def test_query_returns_fallback_for_irrelevant_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(answerer, "_retriever", EmptyRetriever())
    monkeypatch.setattr(answerer, "get_llm", lambda: FallbackLLM())
    _ensure_core_index_stub()
    from apps.api.main import app  # noqa: PLC0415

    client = TestClient(app)
    response = client.post(
        "/query",
        json={"q": "What hue is the quantum banana of Atlantis?", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "Not enough information to answer accurately." in payload.get("answer", "")
    assert payload.get("contexts") == []
