from __future__ import annotations

import logging
import os
from typing import Any, Callable

import structlog
from structlog.contextvars import merge_contextvars

_REDACT_REPLACEMENT = "***REDACTED***"
_DEFAULT_REDACT_FIELDS = {"authorization", "access_token", "refresh_token"}


def _log_level_from_env() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if not level_name:
        return logging.INFO
    return getattr(logging, level_name, logging.INFO)


def _redaction_processor(redact_fields: set[str]) -> Callable[[Any, str, dict[str, Any]], dict[str, Any]]:
    lower_fields = {field.lower() for field in redact_fields}

    def processor(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        for key in list(event_dict.keys()):
            if isinstance(key, str) and key.lower() in lower_fields:
                event_dict[key] = _REDACT_REPLACEMENT
        return event_dict

    return processor


def _redact_fields_from_env() -> set[str]:
    raw = os.getenv("LOG_REDACT_FIELDS", "")
    if not raw:
        return set(_DEFAULT_REDACT_FIELDS)
    fields = {item.strip() for item in raw.split(",") if item.strip()}
    return set(_DEFAULT_REDACT_FIELDS).union(fields)


def configure_logging() -> None:
    """Configure structlog with JSON output, contextvars, and redaction."""

    level = _log_level_from_env()
    redact_fields = _redact_fields_from_env()
    redactor = _redaction_processor(redact_fields)

    logging.basicConfig(level=level, format="%(message)s", force=True)

    processors = [
        merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        redactor,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )
