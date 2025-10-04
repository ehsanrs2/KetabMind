from __future__ import annotations

import importlib
from pathlib import Path


def test_index_and_search_offline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    module = importlib.import_module("backend.local_qdrant")
    module = importlib.reload(module)

    collection = "offline-test"
    pdf_path = Path("docs/fixtures/sample.pdf")

    chunk_ids = module.index_path(pdf_path, collection=collection)
    assert chunk_ids, "indexing should return chunk identifiers"

    results = module.search("Sample PDF", collection=collection, top_k=1)
    assert results, "search should return at least one hit"
    payload = results[0]["payload"]
    assert "Sample" in payload["text"]
    assert payload["book_id"] == pdf_path.stem

    storage = Path(tmp_path, ".ketabmind", "qdrant")
    assert storage.exists(), "local storage directory should be created"
