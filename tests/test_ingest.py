from pathlib import Path

import fitz

from core.ingest.pdf import extract_text


def test_extract_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World")
    doc.save(pdf_path)
    doc.close()
    lines = list(extract_text(pdf_path))
    assert "Hello World" in lines[0]
