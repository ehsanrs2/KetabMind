"""PDF ingestion using PyMuPDF."""

from pathlib import Path
from typing import Iterable

import fitz


def extract_text(path: Path) -> Iterable[str]:
    """Yield pages of text from a PDF file."""
    with fitz.open(path) as doc:
        for page in doc:
            yield page.get_text()
