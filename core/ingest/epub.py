"""EPUB ingestion using ebooklib."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, TypedDict

from ebooklib import epub


class PageDict(TypedDict):
    page_num: int
    text: str
    section: Optional[str]


def _html_to_text(html: str) -> str:
    """Strip HTML tags to plain text."""
    text = re.sub("<[^>]+>", "", html)
    return text.strip()


def extract_pages(path: Path, chars: int = 1024) -> Iterable[PageDict]:
    """Yield plain-text pages from an EPUB."""
    book = epub.read_epub(str(path))
    docs = [book.get_item_with_id(idref) for idref, _ in book.spine]
    buffer = ""
    page_num = 1
    for item in docs:
        html = item.get_content().decode("utf-8", errors="ignore")
        buffer += _html_to_text(html) + "\n"
        while len(buffer) >= chars:
            yield {
                "page_num": page_num,
                "text": buffer[:chars].strip(),
                "section": None,
            }
            page_num += 1
            buffer = buffer[chars:]
    if buffer.strip():
        yield {"page_num": page_num, "text": buffer.strip(), "section": None}
