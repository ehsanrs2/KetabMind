# tests/helpers.py
from __future__ import annotations

import sys
import time
import types
from collections.abc import Callable
from importlib.machinery import ModuleSpec
from typing import Any, cast

import pytest


def setup_api_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install lightweight runtime stubs for optional deps used in tests.

    Design goals:
    - Prefer real libraries if available. Only provide stubs when imports fail.
    - Keep runtime surface minimal but compatible with code paths used in tests.
    - Provide enough typing hints to keep mypy quiet without broad ignores.
    """

    # --- limits --------------------------------------------------------------
    if "limits" not in sys.modules:
        limits_stub = types.ModuleType("limits")

        class RateLimitItemPerSecond:
            """Minimal stand-in for limits.RateLimitItemPerSecond."""

            def __init__(self, amount: int, multiples: int) -> None:
                self.amount = amount
                self.multiples = multiples

        cast(Any, limits_stub).RateLimitItemPerSecond = RateLimitItemPerSecond
        monkeypatch.setitem(sys.modules, "limits", limits_stub)

    # --- slowapi -------------------------------------------------------------
    if "slowapi" not in sys.modules:
        slowapi_stub = types.ModuleType("slowapi")

        class _LimiterBackend:
            """Backend with permissive behavior for tests."""

            def test(self, *args: Any, **kwargs: Any) -> bool:
                return True

            def hit(self, *args: Any, **kwargs: Any) -> None:
                return None

            def get_window_stats(self, *args: Any, **kwargs: Any) -> tuple[float, int]:
                return (time.time(), 0)

        class Limiter:
            """Minimal stand-in for slowapi.Limiter."""

            def __init__(
                self,
                key_func: Callable[[Any], str] | None = None,
                *,
                enabled: bool = False,
                **kwargs: Any,
            ) -> None:
                self._key_func = key_func
                self.enabled = enabled
                self.limiter = _LimiterBackend()

        cast(Any, slowapi_stub).Limiter = Limiter
        monkeypatch.setitem(sys.modules, "slowapi", slowapi_stub)

    # --- prometheus_client ---------------------------------------------------
    if "prometheus_client" not in sys.modules:
        prom_stub = types.ModuleType("prometheus_client")

        class _Metric:
            """Counter/Gauge/Histogram-compatible test metric."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                return None

            def labels(self, *args: Any, **kwargs: Any) -> _Metric:
                return self

            def inc(self, *args: Any, **kwargs: Any) -> None:
                return None

            def observe(self, *args: Any, **kwargs: Any) -> None:
                return None

            def set(self, *args: Any, **kwargs: Any) -> None:
                return None

        cast(Any, prom_stub).CONTENT_TYPE_LATEST = "text/plain"
        cast(Any, prom_stub).Counter = _Metric
        cast(Any, prom_stub).Gauge = _Metric
        cast(Any, prom_stub).Histogram = _Metric

        class _Registry:
            def __init__(self) -> None:
                self._names_to_collectors: dict[str, Any] = {}

        cast(Any, prom_stub).REGISTRY = _Registry()

        def generate_latest() -> bytes:
            return b""

        cast(Any, prom_stub).generate_latest = generate_latest
        monkeypatch.setitem(sys.modules, "prometheus_client", prom_stub)

    # --- numpy (prefer real; only stub if missing) --------------------------
    # If a previous stub without 'ndarray' slipped in, drop it first.
    if "numpy" in sys.modules and not hasattr(sys.modules["numpy"], "ndarray"):
        sys.modules.pop("numpy", None)

    try:
        # Try to import real numpy; do NOT override if it works.
        import numpy as _np  # noqa: F401
    except Exception:
        numpy_stub = types.ModuleType("numpy")

        # Minimal runtime shims so torch's collate can register on np.ndarray.
        class ndarray:
            """Minimal ndarray placeholder for isinstance checks."""

            # Optional: keep a simple shape/dtype surface if needed in tests
            shape: tuple[int, ...] = ()
            dtype: Any = None

        class memmap(ndarray):
            """memmap is a subclass of ndarray in real numpy."""

            pass

        def asarray(data: Any, dtype: Any = None) -> Any:
            return data

        class _Floating(float):
            """Typing-friendly float subclass with __class_getitem__."""

            @classmethod
            def __class_getitem__(cls, item: Any) -> Any:
                return float

        # Provide attributes used across code/tests/libraries.
        cast(Any, numpy_stub).ndarray = ndarray
        cast(Any, numpy_stub).memmap = memmap
        cast(Any, numpy_stub).asarray = asarray
        cast(Any, numpy_stub).float32 = _Floating
        cast(Any, numpy_stub).floating = _Floating
        numpy_stub.__spec__ = ModuleSpec("numpy", loader=None)
        monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

        # numpy.typing stub (only if real numpy.typing not present)
        if "numpy.typing" not in sys.modules:
            typing_stub = types.ModuleType("numpy.typing")
            # Use a very permissive alias for type checkers when numpy is absent
            from typing import Any as _NDArray

            cast(Any, typing_stub).NDArray = _NDArray
            typing_stub.__spec__ = ModuleSpec("numpy.typing", loader=None)
            monkeypatch.setitem(sys.modules, "numpy.typing", typing_stub)

    # --- typer ---------------------------------------------------------------
    if "typer" not in sys.modules:
        typer_stub = types.ModuleType("typer")

        class _Typer:
            """Minimal Typer app that returns no-op decorators."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                return None

            def command(
                self, *args: Any, **kwargs: Any
            ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                    return func

                return decorator

            def __call__(self, *args: Any, **kwargs: Any) -> None:
                return None

        def option(default: Any, *args: Any, **kwargs: Any) -> Any:
            return default

        cast(Any, typer_stub).Typer = _Typer
        cast(Any, typer_stub).Option = option
        monkeypatch.setitem(sys.modules, "typer", typer_stub)

    # --- pypdf ---------------------------------------------------------------
    if "pypdf" not in sys.modules:
        pypdf_stub = types.ModuleType("pypdf")

        class _PdfPage:
            def __init__(self, text: str) -> None:
                self._text = text

            def extract_text(self) -> str:
                return self._text

        class PdfReader:
            """Very small reader that treats the input file as plain text."""

            def __init__(self, path: str) -> None:
                from pathlib import Path

                try:
                    content = Path(path).read_text(encoding="utf-8")
                except FileNotFoundError:
                    content = ""
                self.pages = [_PdfPage(content)]

        cast(Any, pypdf_stub).PdfReader = PdfReader
        monkeypatch.setitem(sys.modules, "pypdf", pypdf_stub)

    # --- fitz (PyMuPDF) -----------------------------------------------------
    if "fitz" not in sys.modules:
        fitz_stub = types.ModuleType("fitz")

        class _StubPixmap:
            def tobytes(self, *args: Any, **kwargs: Any) -> bytes:
                return b""

        class _StubPage:
            def get_text(self) -> str:
                return ""

            def get_pixmap(self, *args: Any, **kwargs: Any) -> _StubPixmap:
                return _StubPixmap()

        class _StubDocument:
            def __init__(self, path: Any = None) -> None:
                self.page_count = 0

            def load_page(self, index: int) -> _StubPage:
                return _StubPage()

            def close(self) -> None:
                return None

        def open_document(path: Any) -> _StubDocument:
            return _StubDocument(path)

        cast(Any, fitz_stub).open = open_document
        cast(Any, fitz_stub).Document = _StubDocument
        cast(Any, fitz_stub).Pixmap = _StubPixmap
        monkeypatch.setitem(sys.modules, "fitz", fitz_stub)
