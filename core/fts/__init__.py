"""Full-text search backend management."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.config import settings


@dataclass(frozen=True)
class FTSMatch:
    """Result returned by FTS backends."""

    book_id: str
    page_num: int
    text: str
    score: float


class FTSBackendProto:
    """Protocol-like base class for FTS backends."""

    available: bool = False

    def is_available(self) -> bool:
        return bool(self.available)

    def index_book(
        self,
        book_id: str,
        pages: Iterable[tuple[int, str]],
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def search(
        self,
        query: str,
        *,
        book_id: str | None = None,
        limit: int | None = None,
    ) -> Sequence[FTSMatch]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        return None


class _NullBackend(FTSBackendProto):
    """Placeholder backend when FTS is disabled."""

    available = False

    def index_book(self, book_id: str, pages: Iterable[tuple[int, str]]) -> None:
        return None

    def search(
        self,
        query: str,
        *,
        book_id: str | None = None,
        limit: int | None = None,
    ) -> Sequence[FTSMatch]:
        return []


_cached_backend: FTSBackendProto | None = None
_cached_signature: tuple[str, str] | None = None


def _backend_signature() -> tuple[str, str]:
    backend = (settings.fts_backend or "").strip().lower()
    path = settings.fts_sqlite_path if hasattr(settings, "fts_sqlite_path") else ""
    return backend, str(path or "")


def _build_backend() -> FTSBackendProto:
    backend, path = _backend_signature()
    if backend == "sqlite":
        from .sqlite import SqliteFTSBackend

        return SqliteFTSBackend(Path(path))
    return _NullBackend()


def reset_backend() -> None:
    """Reset the cached backend (primarily for tests)."""

    global _cached_backend, _cached_signature
    if _cached_backend is not None:
        _cached_backend.close()
    _cached_backend = None
    _cached_signature = None


def get_backend() -> FTSBackendProto:
    """Return the configured FTS backend."""

    global _cached_backend, _cached_signature
    signature = _backend_signature()
    if _cached_backend is None or signature != _cached_signature:
        if _cached_backend is not None:
            _cached_backend.close()
        _cached_backend = _build_backend()
        _cached_signature = signature
    return _cached_backend


__all__ = ["FTSMatch", "FTSBackendProto", "get_backend", "reset_backend"]
