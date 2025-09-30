from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from . import contextvars

__all__ = [
    "configure",
    "get_logger",
    "make_filtering_bound_logger",
    "processors",
    "contextvars",
]


@dataclass
class _Logger:
    name: str

    def bind(self, **_: Any) -> "_Logger":  # pragma: no cover - compatibility helper
        return self

    def _log(self, level: str, event: str, **kwargs: Any) -> None:  # pragma: no cover - logging helper
        message = f"[{level}] {self.name}: {event}"
        if kwargs:
            message += " " + " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        logging.getLogger(self.name).log(getattr(logging, level), message)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log("ERROR", event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        self._log("ERROR", event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log("DEBUG", event, **kwargs)


class processors:  # pragma: no cover - minimal stubs
    class TimeStamper:
        def __init__(self, fmt: str = "iso") -> None:
            self.fmt = fmt

        def __call__(self, logger: Any, name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
            return event_dict

    @staticmethod
    def add_log_level(logger: Any, name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        event_dict.setdefault("level", name)
        return event_dict

    class StackInfoRenderer:
        def __call__(self, logger: Any, name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
            return event_dict

    @staticmethod
    def format_exc_info(logger: Any, name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        return event_dict

    class JSONRenderer:
        def __call__(self, logger: Any, name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
            return event_dict


def configure(*_, **__) -> None:  # pragma: no cover - configuration stub
    return None


def make_filtering_bound_logger(level: int) -> Callable[..., _Logger]:  # pragma: no cover - stub
    def factory(*args: Any, **kwargs: Any) -> _Logger:
        name = args[0] if args else kwargs.get("name", "structlog")
        return _Logger(str(name))

    return factory


def get_logger(name: str | None = None) -> _Logger:
    return _Logger(name or "structlog")
