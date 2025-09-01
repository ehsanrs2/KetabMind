from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    book_id: str
    page_start: int
    page_end: int
    chunk_id: str


def sliding_window_chunks(
    texts_pages: Iterable[tuple[str, int]], book_id: str, size: int, overlap: int
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[tuple[str, int]] = []
    for text, page in texts_pages:
        buf.append((text, page))
    content = []
    pages = []
    for t, p in buf:
        content.append(t)
        pages.append(p)
    full = "\n".join(content)
    step = max(1, size - overlap)
    for i in range(0, max(1, len(full)), step):
        window = full[i : i + size]
        if not window.strip():
            continue
        page_start = min(pages) if pages else -1
        page_end = max(pages) if pages else -1
        chunk_id = f"{book_id}:{i}:{i+len(window)}"
        chunks.append(
            Chunk(
                text=window,
                book_id=book_id,
                page_start=page_start,
                page_end=page_end,
                chunk_id=chunk_id,
            )
        )
    return chunks
