from __future__ import annotations

import json
import logging

import pytest

import structlog
from apps.api.middleware import RequestIDMiddleware
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from utils.logging import configure_logging


@pytest.fixture
def configured_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LOG_REDACT_FIELDS", "authorization,token")
    configure_logging()


def _collect_json_logs(records: list[logging.LogRecord]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for record in records:
        message = record.getMessage()
        try:
            events.append(json.loads(message))
        except json.JSONDecodeError:
            continue
    return events


def test_request_logging_includes_request_id_and_redaction(
    configured_logging: None, caplog: pytest.LogCaptureFixture
) -> None:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)
    log = structlog.get_logger("test.logging")

    @app.get("/ping")
    def ping(_request: Request) -> dict[str, str]:
        log.info("test.event", authorization="secret", token="abc123")  # noqa: S106
        return {"status": "ok"}

    client = TestClient(app)
    with caplog.at_level(logging.INFO):
        response = client.get("/ping", headers={"X-Request-ID": "req-123"})

    assert response.status_code == 200

    events = _collect_json_logs(caplog.records)
    assert events, "Expected structured log events"

    request_log = next(event for event in events if event.get("event") == "http.request")
    assert request_log["request_id"] == "req-123"
    assert request_log["path"] == "/ping"
    assert request_log["method"] == "GET"
    assert request_log["status"] == 200
    assert isinstance(request_log["latency_ms"], float)

    handler_log = next(event for event in events if event.get("event") == "test.event")
    assert handler_log["request_id"] == "req-123"
    assert handler_log["path"] == "/ping"
    assert handler_log["method"] == "GET"
    assert "status" in handler_log
    assert "latency_ms" in handler_log
    assert handler_log["authorization"] == "***REDACTED***"
    assert handler_log["token"] == "***REDACTED***"  # noqa: S105
