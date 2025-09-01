from pathlib import Path

from core.ingest.epub import extract_pages


def test_extract_epub() -> None:
    epub_path = Path("docs/fixtures/sample.epub")
    pages = list(extract_pages(epub_path))
    assert any("Hello EPUB world." in p["text"] for p in pages)
