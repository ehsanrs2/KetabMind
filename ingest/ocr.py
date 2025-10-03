"""Tesseract OCR helpers for Farsi text extraction."""

from __future__ import annotations

import os
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def _tessdata_prefix() -> str | None:
    """Return the configured tessdata directory if provided."""

    prefix = os.getenv("TESSDATA_PREFIX")
    if prefix:
        log.debug("ocr.tessdata_prefix", path=prefix)
    return prefix


def image_to_text(image: Any) -> str:
    """Extract Farsi text from an image using Tesseract."""

    import pytesseract  # imported lazily for easier testing/mocking

    tessdata_prefix = _tessdata_prefix()
    kwargs: dict[str, Any] = {"lang": "fas"}
    if tessdata_prefix:
        kwargs["config"] = f'--tessdata-dir "{tessdata_prefix}"'
    return pytesseract.image_to_string(image, **kwargs)


__all__ = ["image_to_text"]
