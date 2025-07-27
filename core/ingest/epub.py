"""EPUB ingestion using Apache Tika."""

from pathlib import Path
from typing import Iterable

from tika import parser


def extract_text(path: Path) -> Iterable[str]:
    """Yield full text from an EPUB file."""
    parsed = parser.from_file(str(path))
    content = parsed.get("content", "")
    for line in content.splitlines():
        if line.strip():
            yield line
