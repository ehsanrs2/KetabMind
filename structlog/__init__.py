from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from . import contextvars

__all__ = [
    "configure",
    "get_logger",
    "make_filtering_bound_logger",
    "processors",
    "contextvars",
]


_processors: list[Callable[[Any, str, dict[str, Any]], Any]] = []
_wrapper_factory: Callable[..., _Logger] | None = None


@dataclass
class _Logger:
    name: str

    def bind(self, **_: Any) -> _Logger:  # pragma: no cover - logging helper
        return self

    def _run_processors(self, level: str, event: str, event_dict: dict[str, Any]) -> Any:
        current = event_dict
        for processor in _processors:
            current = processor(self, level, current)
        return current

    def _log(self, level: str, event: str, **kwargs: Any) -> None:
        event_dict: dict[str, Any] = {"event": event, **kwargs}
        processed = self._run_processors(level, event, event_dict)
        message = json.dumps(processed) if isinstance(processed, Mapping) else str(processed)
        logging.getLogger(self.name).log(getattr(logging, level), message)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log("ERROR", event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self._log("ERROR", event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._log("DEBUG", event, **kwargs)


class processors:  # pragma: no cover - minimal implementations
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
        if event_dict.get("exc_info") and isinstance(event_dict["exc_info"], bool):
            event_dict.pop("exc_info", None)
        return event_dict

    class JSONRenderer:
        def __call__(self, logger: Any, name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
            return event_dict


def configure(
    *,
    processors: Iterable[Callable[[Any, str, dict[str, Any]], Any]] | None = None,
    wrapper_class: Callable[..., _Logger] | None = None,
    logger_factory: Callable[..., Any] | None = None,
    cache_logger_on_first_use: bool | None = None,
) -> None:
    del logger_factory, cache_logger_on_first_use
    global _processors, _wrapper_factory
    _processors = list(processors or [])
    _wrapper_factory = wrapper_class


def make_filtering_bound_logger(
    level: int,
) -> Callable[..., _Logger]:  # pragma: no cover - level unused
    def factory(name: str | None = None, **_: Any) -> _Logger:
        return _Logger(name or "structlog")

    return factory


def get_logger(name: str | None = None) -> _Logger:
    if _wrapper_factory is None:
        return _Logger(name or "structlog")
    return _wrapper_factory(name=name or "structlog")
