from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

# Provide lightweight stubs for optional runtime dependencies before
# importing the module under test.
typer_module = types.ModuleType("typer")


def _typer_option(
    default: object = None, *args: object, **kwargs: object
) -> object:  # pragma: no cover - stub
    return default


class _TyperApp:
    def command(self, *args: object, **kwargs: object):  # pragma: no cover - stub
        def decorator(func):
            return func

        return decorator


typer_module.Option = _typer_option
typer_module.Typer = lambda *args, **kwargs: _TyperApp()  # pragma: no cover - stub
sys.modules.setdefault("typer", typer_module)

structlog_module = types.ModuleType("structlog")


class _DummyLogger:
    def __getattr__(self, _name: str):  # pragma: no cover - stub
        return lambda *args, **kwargs: None


structlog_module.get_logger = lambda *args, **kwargs: _DummyLogger()  # pragma: no cover - stub
sys.modules.setdefault("structlog", structlog_module)

fitz_stub = types.SimpleNamespace()
fitz_stub.open = lambda _path: None
sys.modules.setdefault("fitz", fitz_stub)

pil_module = types.ModuleType("PIL")
image_module = types.ModuleType("Image")
image_module.open = lambda data: data  # pragma: no cover - passthrough for stub
pil_module.Image = image_module
sys.modules.setdefault("PIL", pil_module)
sys.modules.setdefault("PIL.Image", image_module)

pypdf_module = types.ModuleType("pypdf")


class _DummyPdfReader:
    def __init__(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - stub
        self.pages = []


pypdf_module.PdfReader = _DummyPdfReader
sys.modules.setdefault("pypdf", pypdf_module)

pdf_reader = importlib.import_module("ingest.pdf_reader")


class DummyDoc:
    def __init__(self, pages: list[object]):
        self._pages = pages
        self.page_count = len(pages)
        self.closed = False

    def load_page(self, index: int) -> object:
        return self._pages[index]

    def close(self) -> None:
        self.closed = True


class DummyPage:
    def __init__(self, text: str, pixmap_factory: Callable[[], object] | None = None):
        self._text = text
        self._pixmap_factory = pixmap_factory or (lambda: object())

    def get_text(self) -> str:
        return self._text

    def get_pixmap(self, **_kwargs: object) -> object:
        return self._pixmap_factory()


def test_textual_pdf_skips_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [DummyPage("hello"), DummyPage("world")]
    doc = DummyDoc(pages)

    monkeypatch.setattr(fitz_stub, "open", lambda _path: doc)
    monkeypatch.setenv("OCR_FA", "true")

    ocr_calls: list[object] = []
    monkeypatch.setattr(
        pdf_reader, "image_to_text", lambda image: ocr_calls.append(image) or "ignored"
    )

    result = pdf_reader.read_pdf(Path("textual.pdf"))

    assert [page.text for page in result] == ["hello", "world"]
    assert doc.closed is True
    assert ocr_calls == []


class DummyPixmap:
    def __init__(self, marker: str):
        self.marker = marker


def test_scanned_pdf_uses_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        DummyPage("", pixmap_factory=lambda: DummyPixmap("p1")),
        DummyPage(" ", pixmap_factory=lambda: DummyPixmap("p2")),
    ]
    doc = DummyDoc(pages)

    monkeypatch.setattr(fitz_stub, "open", lambda _path: doc)
    monkeypatch.setenv("OCR_FA", "true")

    images: list[str] = []
    monkeypatch.setattr(
        pdf_reader, "_pixmap_to_image", lambda pix: images.append(pix.marker) or pix.marker
    )

    ocr_texts = {"p1": "page one", "p2": "page two"}

    def fake_ocr(image: str) -> str:
        return ocr_texts[image]

    monkeypatch.setattr(pdf_reader, "image_to_text", fake_ocr)

    result = pdf_reader.read_pdf(Path("scanned.pdf"))

    assert [page.text for page in result] == ["page one", "page two"]
    assert images == ["p1", "p2"]
    assert doc.closed is True
