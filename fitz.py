"""Lightweight stub of the PyMuPDF `fitz` module for tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Pixmap:
    """Minimal pixmap stub returning empty PNG bytes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial stub
        return None

    def tobytes(self, *args: Any, **kwargs: Any) -> bytes:
        return b""


@dataclass
class _StubPage:
    """Single PDF page stub."""

    def get_text(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def get_pixmap(self, *args: Any, **kwargs: Any) -> Pixmap:
        return Pixmap()


class Document:
    """Simplified document that exposes PyMuPDF-like API."""

    def __init__(self, _path: Any = None) -> None:
        self.page_count = 0

    def load_page(self, _index: int) -> _StubPage:
        return _StubPage()

    def close(self) -> None:  # pragma: no cover - trivial stub
        return None


def open(_path: Any = None) -> Document:
    """Return a stub document for the provided path."""

    return Document(_path)


__all__ = ["Document", "Pixmap", "open"]
