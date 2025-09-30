from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def bind_contextvars(**kwargs: Any) -> None:  # pragma: no cover - simple stub
    if "request_id" in kwargs:
        _request_id.set(str(kwargs["request_id"]))


def clear_contextvars() -> None:  # pragma: no cover - simple stub
    _request_id.set(None)


def merge_contextvars(event_dict: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - simple stub
    request_id = _request_id.get()
    if request_id is not None:
        event_dict.setdefault("request_id", request_id)
    return event_dict
