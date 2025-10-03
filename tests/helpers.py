from __future__ import annotations

import sys
import time
import types
from typing import Any

import pytest


def setup_api_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    if "limits" not in sys.modules:
        limits_stub = types.ModuleType("limits")

        class RateLimitItemPerSecond:  # type: ignore[too-few-public-methods]
            def __init__(self, amount: int, multiples: int) -> None:
                self.amount = amount
                self.multiples = multiples

        limits_stub.RateLimitItemPerSecond = RateLimitItemPerSecond
        monkeypatch.setitem(sys.modules, "limits", limits_stub)

    if "slowapi" not in sys.modules:
        slowapi_stub = types.ModuleType("slowapi")

        class _LimiterBackend:  # type: ignore[too-few-public-methods]
            def test(self, *args: Any, **kwargs: Any) -> bool:
                return True

            def hit(self, *args: Any, **kwargs: Any) -> None:
                return None

            def get_window_stats(self, *args: Any, **kwargs: Any) -> tuple[float, int]:
                return (time.time(), 0)

        class Limiter:  # type: ignore[too-few-public-methods]
            def __init__(self, key_func=None, *, enabled: bool = False, **kwargs: Any) -> None:
                self._key_func = key_func
                self.enabled = enabled
                self.limiter = _LimiterBackend()

        slowapi_stub.Limiter = Limiter
        monkeypatch.setitem(sys.modules, "slowapi", slowapi_stub)

    if "prometheus_client" not in sys.modules:
        prom_stub = types.ModuleType("prometheus_client")

        class _Metric:  # type: ignore[too-few-public-methods]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                return None

            def labels(self, *args: Any, **kwargs: Any) -> "_Metric":
                return self

            def inc(self, *args: Any, **kwargs: Any) -> None:
                return None

            def observe(self, *args: Any, **kwargs: Any) -> None:
                return None

            def set(self, *args: Any, **kwargs: Any) -> None:
                return None

        prom_stub.CONTENT_TYPE_LATEST = "text/plain"
        prom_stub.Counter = _Metric
        prom_stub.Gauge = _Metric
        prom_stub.Histogram = _Metric

        def generate_latest() -> bytes:
            return b""

        prom_stub.generate_latest = generate_latest
        monkeypatch.setitem(sys.modules, "prometheus_client", prom_stub)

    if "numpy" not in sys.modules:
        numpy_stub = types.ModuleType("numpy")

        def asarray(data: Any, dtype: Any = None) -> Any:
            return data

        class _Floating(float):  # type: ignore[too-few-public-methods]
            @classmethod
            def __class_getitem__(cls, item: Any) -> Any:
                return float

        numpy_stub.asarray = asarray
        numpy_stub.float32 = _Floating
        numpy_stub.floating = _Floating
        from importlib.machinery import ModuleSpec

        numpy_stub.__spec__ = ModuleSpec("numpy", loader=None)
        monkeypatch.setitem(sys.modules, "numpy", numpy_stub)
        typing_stub = types.ModuleType("numpy.typing")
        typing_stub.NDArray = list  # type: ignore[assignment]
        typing_stub.__spec__ = ModuleSpec("numpy.typing", loader=None)
        monkeypatch.setitem(sys.modules, "numpy.typing", typing_stub)

    if "typer" not in sys.modules:
        typer_stub = types.ModuleType("typer")

        class _Typer:  # type: ignore[too-few-public-methods]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                return None

            def command(self, *args: Any, **kwargs: Any):
                def decorator(func):
                    return func

                return decorator

            def __call__(self, *args: Any, **kwargs: Any) -> None:
                return None

        def option(default: Any, *args: Any, **kwargs: Any) -> Any:
            return default

        typer_stub.Typer = _Typer
        typer_stub.Option = option
        monkeypatch.setitem(sys.modules, "typer", typer_stub)

    if "pypdf" not in sys.modules:
        pypdf_stub = types.ModuleType("pypdf")

        class _PdfPage:  # type: ignore[too-few-public-methods]
            def __init__(self, text: str) -> None:
                self._text = text

            def extract_text(self) -> str:
                return self._text

        class PdfReader:  # type: ignore[too-few-public-methods]
            def __init__(self, path: str) -> None:
                from pathlib import Path

                try:
                    content = Path(path).read_text(encoding="utf-8")
                except FileNotFoundError:
                    content = ""
                self.pages = [_PdfPage(content)]

        pypdf_stub.PdfReader = PdfReader
        monkeypatch.setitem(sys.modules, "pypdf", pypdf_stub)

    if "fitz" not in sys.modules:
        fitz_stub = types.ModuleType("fitz")

        class _StubPixmap:  # type: ignore[too-few-public-methods]
            def tobytes(self, *args: Any, **kwargs: Any) -> bytes:
                return b""

        class _StubPage:  # type: ignore[too-few-public-methods]
            def get_text(self) -> str:
                return ""

            def get_pixmap(self, *args: Any, **kwargs: Any) -> _StubPixmap:
                return _StubPixmap()

        class _StubDocument:  # type: ignore[too-few-public-methods]
            def __init__(self, path: Any = None) -> None:
                self.page_count = 0

            def load_page(self, index: int) -> _StubPage:
                return _StubPage()

            def close(self) -> None:
                return None

        def open_document(path: Any) -> _StubDocument:
            return _StubDocument(path)

        fitz_stub.open = open_document
        fitz_stub.Document = _StubDocument
        fitz_stub.Pixmap = _StubPixmap
        monkeypatch.setitem(sys.modules, "fitz", fitz_stub)
