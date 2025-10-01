from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_context: ContextVar[dict[str, Any]] = ContextVar("structlog_context", default={})


def bind_contextvars(**kwargs: Any) -> None:
    current = dict(_context.get())
    current.update(kwargs)
    _context.set(current)


def clear_contextvars() -> None:
    _context.set({})


def merge_contextvars(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    context = _context.get()
    merged = dict(context)
    merged.update(event_dict)
    return merged
