from __future__ import annotations

import logging
import os
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog
from structlog.contextvars import merge_contextvars

_REDACT_REPLACEMENT = "***REDACTED***"
_DEFAULT_REDACT_FIELDS = {"authorization", "access_token", "refresh_token"}


def _log_level_from_env() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if not level_name:
        return logging.INFO
    return getattr(logging, level_name, logging.INFO)


def _redaction_processor(
    redact_fields: set[str],
) -> Callable[[Any, str, dict[str, Any]], dict[str, Any]]:
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


def _log_destination() -> Path | None:
    explicit = os.getenv("LOG_FILE")
    if explicit:
        normalized = explicit.strip()
        if normalized.lower() == "stdout":
            return None
        return Path(normalized)

    directory = os.getenv("LOG_DIR", "logs").strip()
    if not directory:
        return None
    return Path(directory) / "ketabmind.log"


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        return default


def _build_handlers(level: int) -> list[logging.Handler]:
    formatter = logging.Formatter("%(message)s")
    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    destination = _log_destination()
    if destination is not None:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            max_bytes = _int_from_env("LOG_FILE_MAX_BYTES", 10_000_000)
            backup_count = _int_from_env("LOG_FILE_BACKUP_COUNT", 5)
            file_handler = RotatingFileHandler(
                destination,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except OSError:
            # Fall back to stdout-only logging if file handler cannot be created.
            pass

    return handlers


def configure_logging() -> None:
    """Configure structlog with JSON output, contextvars, and redaction."""

    level = _log_level_from_env()
    redact_fields = _redact_fields_from_env()
    redactor = _redaction_processor(redact_fields)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=_build_handlers(level),
        force=True,
    )

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
