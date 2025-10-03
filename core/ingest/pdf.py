"""PDF ingestion using PyPDF with lightweight TOC support."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterable as TypingIterable
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader

from ..config import settings


class PageDict(TypedDict):
    """Output schema for extracted pages."""

    page_num: int
    text: str
    section: str | None


def _load_toc(reader: PdfReader) -> list[tuple[int, str]]:
    """Return a sorted list of (page_number, title) outlines."""

    try:
        outlines: TypingIterable[object] = getattr(reader, "outline", reader.outlines)
    except AttributeError:  # pragma: no cover - defensive for older pypdf
        outlines = []

    entries: list[tuple[int, str]] = []

    def _walk(items: TypingIterable[object]) -> None:
        for item in items:
            if isinstance(item, list):  # nested outline list
                _walk(item)
                continue
            title = getattr(item, "title", None)
            if not title:
                continue
            try:
                page_index = reader.get_destination_page_number(item)
            except Exception:  # pragma: no cover - fallback when destination missing
                continue
            entries.append((page_index + 1, title))

    if outlines:
        try:
            _walk(outlines)
        except Exception:  # pragma: no cover - unexpected outline structure
            entries.clear()

    entries.sort(key=lambda item: item[0])
    return entries


def extract_pages(
    path: Path,
    *,
    lines_top: int | None = None,
    lines_bottom: int | None = None,
) -> Iterable[PageDict]:
    """Yield cleaned pages from a PDF using pypdf only."""

    top = settings.ingest_header_lines if lines_top is None else lines_top
    bottom = settings.ingest_footer_lines if lines_bottom is None else lines_bottom

    reader = PdfReader(str(path))
    toc = _load_toc(reader)
    toc_idx = 0
    current: str | None = None

    for i, page in enumerate(reader.pages, start=1):
        while toc_idx < len(toc) and toc[toc_idx][0] <= i:
            current = toc[toc_idx][1]
            toc_idx += 1
        text = page.extract_text() or ""
        lines = text.splitlines()
        if top:
            lines = lines[top:]
        if bottom:
            lines = lines[:-bottom] if bottom < len(lines) else []
        cleaned = "\n".join(lines).strip()
        yield {"page_num": i, "text": cleaned, "section": current}
