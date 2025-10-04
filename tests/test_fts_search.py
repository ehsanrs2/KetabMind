from __future__ import annotations

import json
from pathlib import Path

from core.fts.sqlite import SqliteFTSBackend


def _load_sample_book() -> list[tuple[int, str]]:
    data_path = Path(__file__).parent / "data" / "sample_book.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    pages: list[tuple[int, str]] = []
    for entry in payload:
        page = int(entry["page"])
        text = str(entry["text"])
        pages.append((page, text))
    return pages


def test_sqlite_fts_finds_expected_page(tmp_path: Path) -> None:
    backend_path = tmp_path / "fts.db"
    backend = SqliteFTSBackend(backend_path)
    try:
        pages = _load_sample_book()
        backend.index_book("sample-book", pages)

        matches = backend.search("luminous horizon", book_id="sample-book", limit=5)
        page_numbers = [match.page_num for match in matches]

        assert sorted(page_numbers) == [2, 3]
    finally:
        backend.close()
