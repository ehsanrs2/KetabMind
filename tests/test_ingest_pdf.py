from pathlib import Path

from core.ingest.pdf import extract_pages


def test_extract_pdf() -> None:
    pdf_path = Path("docs/fixtures/sample.pdf")
    pages = list(extract_pages(pdf_path))
    assert pages and "Sample PDF page 1" in pages[0]["text"]

