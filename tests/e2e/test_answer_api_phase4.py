from __future__ import annotations

import sys
import types
from typing import Any

import pytest

pytest.importorskip("fastapi")
from core.answer import answerer  # noqa: E402
from fastapi import URL, Request  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    fa_result = {
        "answer": "این یک پاسخ نمونه است.",
        "contexts": [
            {
                "id": "fa-ctx-1",
                "book_id": "ketab-1",
                "page": 13,
                "page_start": 12,
                "page_end": 14,
                "hybrid": 0.68,
                "text": "نمونه متن",
                "metadata": {"version": "v1"},
            }
        ],
        "debug": {
            "stats": {
                "coverage": 0.72,
                "confidence": 0.68,
            }
        },
    }

    en_result = {
        "answer": "This is a sample answer.",
        "contexts": [
            {
                "id": "en-ctx-1",
                "book_id": "ref-book",
                "page": 7,
                "hybrid": 0.44,
                "text": "sample text",
                "metadata": {"version": "v2"},
            }
        ],
        "debug": {
            "stats": {
                "coverage": 0.55,
                "confidence": 0.44,
            }
        },
    }

    responses = {
        "پرسش فارسی": fa_result,
        "english question": en_result,
    }

    def fake_answer(query: str, top_k: int = 8) -> dict:
        try:
            return responses[query]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AssertionError(f"Unexpected query: {query}") from exc

    monkeypatch.setattr(answerer, "answer", fake_answer)
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        types.SimpleNamespace(asarray=lambda data, dtype=None: data),
    )
    index_stub = types.ModuleType("core.index")

    class _IndexResult:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    index_stub.IndexResult = _IndexResult
    index_stub.find_indexed_file = lambda *args, **kwargs: None
    index_stub.index_path = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "core.index", index_stub)

    import apps.api.main as api_main

    monkeypatch.setattr(api_main, "answer", fake_answer)


def test_query_response_meta_and_debug_farsi() -> None:
    from apps.api.main import QueryRequest, query  # noqa: PLC0415

    request = Request("POST", URL("/query"))
    payload = query(
        QueryRequest(q="پرسش فارسی", top_k=1),
        request=request,
        stream=False,
        debug=True,
    )

    assert payload["answer"] == "این یک پاسخ نمونه است."
    assert payload["citations"] == ["[ketab-1:12-14]"]
    assert payload["meta"] == {
        "lang": "fa",
        "coverage": 0.72,
        "confidence": 0.68,
    }

    evidence_map = payload.get("evidence_map")
    assert evidence_map and evidence_map[0]["sentence"] == "این یک پاسخ نمونه است"
    assert evidence_map[0]["supports"] == [{"book_id": "ketab-1", "page": 13, "hybrid": 0.68}]

    used_contexts = payload.get("used_contexts")
    assert used_contexts == [{"id": "fa-ctx-1", "book_id": "ketab-1", "page": 13, "hybrid": 0.68}]


def test_query_response_meta_english() -> None:
    from apps.api.main import QueryRequest, query  # noqa: PLC0415

    request = Request("POST", URL("/query"))
    payload = query(
        QueryRequest(q="english question", top_k=1),
        request=request,
        stream=False,
        debug=False,
    )

    assert payload["answer"] == "This is a sample answer."
    assert payload["citations"] == ["[ref-book:7]"]
    assert payload["meta"] == {
        "lang": "en",
        "coverage": 0.55,
        "confidence": 0.44,
    }
    assert "evidence_map" not in payload
    assert "used_contexts" not in payload
