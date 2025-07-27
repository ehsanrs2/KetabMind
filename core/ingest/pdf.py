"""PDF ingestion using PyMuPDF."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, TypedDict

import fitz

from ..config import settings


class PageDict(TypedDict):
    """Output schema for extracted pages."""

    page_num: int
    text: str
    section: Optional[str]


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
        current: Optional[str] = None
        for i, page in enumerate(doc, start=1):
            while toc_idx < len(toc_sorted) and toc_sorted[toc_idx][2] <= i:
                current = toc_sorted[toc_idx][1]
                toc_idx += 1
            text = page.get_text()
            lines = text.splitlines()
            if top:
                lines = lines[top:]
            if bottom:
                if bottom < len(lines):
                    lines = lines[:-bottom]
                else:
                    lines = []
            cleaned = "\n".join(lines).strip()
            yield {"page_num": i, "text": cleaned, "section": current}
