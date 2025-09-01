from __future__ import annotations

import tempfile
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from core.ingest.pdf_to_text import pdf_to_pages


def _make_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(100, 750, "Sample PDF Page 1: Hello World")
    c.showPage()
    c.drawString(100, 750, "Sample PDF Page 2: Goodbye World")
    c.showPage()
    c.save()


def test_pdf_to_pages_extracts_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "sample.pdf"
        _make_pdf(pdf_path)
        pages = pdf_to_pages(pdf_path)
        assert len(pages) == 2
        assert any("Hello World" in p.text for p in pages)
