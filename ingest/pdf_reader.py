"""PDF reader with OCR fallback for scanned documents."""

from __future__ import annotations

import io
import os
from collections.abc import Iterable
from pathlib import Path

import structlog
from core.ingest.pdf_to_text import Page
from ingest.ocr import image_to_text

try:  # pragma: no cover - runtime dependency validation
    import fitz  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - explicit error when PyMuPDF missing
    raise RuntimeError("PyMuPDF is required for ingest.pdf_reader") from exc

try:  # pragma: no cover - optional dependency for runtime OCR
    from PIL import Image
except ImportError:  # pragma: no cover - defer failure until OCR is required
    Image = None  # type: ignore[assignment]

log = structlog.get_logger(__name__)


def _is_ocr_enabled() -> bool:
    value = os.getenv("OCR_FA", "false").lower()
    return value in {"1", "true", "yes", "on"}


def _pixmap_to_image(pixmap: fitz.Pixmap) -> Image.Image:
    if Image is None:  # pragma: no cover - exercised only when Pillow missing at runtime
        raise RuntimeError("Pillow is required for OCR fallback")
    data = pixmap.tobytes("png")
    image = Image.open(io.BytesIO(data))
    try:
        return image.convert("RGB")
    finally:
        image.load()


def _extract_with_pymupdf(doc: fitz.Document) -> list[tuple[fitz.Page, str]]:
    extracted: list[tuple[fitz.Page, str]] = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        text = (page.get_text() or "").strip()
        extracted.append((page, text))
    return extracted


def _extracted_ratio(pages: Iterable[tuple[object, str]]) -> float:
    items = list(pages)
    if not items:
        return 0.0
    non_empty = sum(1 for _, text in items if text.strip())
    return non_empty / len(items)


def read_pdf(path: Path) -> list[Page]:
    doc = fitz.open(path)
    try:
        extracted = _extract_with_pymupdf(doc)
        ratio = _extracted_ratio(extracted)

        if ratio < 0.1 and _is_ocr_enabled():
            log.info("pdf_reader.ocr_fallback", path=str(path), ratio=ratio)
            pages: list[Page] = []
            for index, (page, _text) in enumerate(extracted, start=1):
                pixmap = page.get_pixmap(alpha=False)
                image = _pixmap_to_image(pixmap)
                ocr_text = image_to_text(image)
                pages.append(Page(page_num=index, text=ocr_text.strip()))
            return pages

        if ratio < 0.1:
            log.info("pdf_reader.ocr_skipped", path=str(path), ratio=ratio)
        else:
            log.debug("pdf_reader.pymupdf_success", path=str(path), ratio=ratio)

        return [
            Page(page_num=index, text=text)
            for index, (_page, text) in enumerate(extracted, start=1)
        ]
    finally:
        doc.close()


__all__ = ["read_pdf"]
