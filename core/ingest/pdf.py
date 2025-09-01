"""PDF ingestion using PyMuPDF."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import fitz

from ..config import settings


class PageDict(TypedDict):
    """Output schema for extracted pages."""

    page_num: int
    text: str
    section: str | None


def extract_pages(
    path: Path,
    *,
    lines_top: int | None = None,
    lines_bottom: int | None = None,
) -> Iterable[PageDict]:
    """Yield cleaned pages from a PDF."""

    top = settings.ingest_header_lines if lines_top is None else lines_top
    bottom = settings.ingest_footer_lines if lines_bottom is None else lines_bottom

    with fitz.open(path) as doc:
        toc = doc.get_toc(simple=True)
        toc_sorted = sorted(toc, key=lambda t: t[2]) if toc else []
        toc_idx = 0
        current: str | None = None
        for i, page in enumerate(doc, start=1):
            while toc_idx < len(toc_sorted) and toc_sorted[toc_idx][2] <= i:
                current = toc_sorted[toc_idx][1]
                toc_idx += 1
            text = page.get_text()
            lines = text.splitlines()
            if top:
                lines = lines[top:]
            if bottom:
                lines = lines[:-bottom] if bottom < len(lines) else []
            cleaned = "\n".join(lines).strip()
            yield {"page_num": i, "text": cleaned, "section": current}
