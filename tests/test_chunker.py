from __future__ import annotations

from core.chunk.chunker import sliding_window_chunks


def test_sliding_window_chunks_basic() -> None:
    texts_pages = [("hello world " * 50, 1), ("second page " * 50, 2)]
    chunks = sliding_window_chunks(texts_pages, book_id="book1", size=100, overlap=20)
    assert chunks, "should produce chunks"
    for ch in chunks:
        assert ch.book_id == "book1"
        assert ch.page_start == 1
        assert ch.page_end == 2
