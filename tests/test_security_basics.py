from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType

import pytest

from fastapi.testclient import TestClient


def _reload_app(
    monkeypatch: pytest.MonkeyPatch, setup: Callable[[], None] | None = None
) -> tuple[ModuleType, ModuleType]:
    """Reload config and app with optional environment setup."""

    import core.config as config

    if setup is not None:
        setup()

    try:
        import prometheus_client
        from prometheus_client import CollectorRegistry

        new_registry = CollectorRegistry()
        prometheus_client.registry.REGISTRY = new_registry
        prometheus_client.REGISTRY = new_registry
        prometheus_client.metrics.REGISTRY = new_registry
    except Exception:  # pragma: no cover - defensive cleanup
        pass

    importlib.reload(config)

    import apps.api.main as api_main

    importlib.reload(api_main)

    return config, api_main


def test_query_rate_limit_sets_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    def _configure_env() -> None:
        monkeypatch.setenv("RATE_LIMIT_QPS", "2")
        monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://allowed.test")

    config, api_main = _reload_app(monkeypatch, _configure_env)

    monkeypatch.setattr(api_main, "answer", lambda *args, **kwargs: {"answer": "ok"})
    monkeypatch.setattr(api_main, "build_query_response", lambda *args, **kwargs: {"answer": "ok"})

    client = TestClient(api_main.app)
    assert client.post("/query", json={"q": "hello"}).status_code == 200
    response = client.post("/query", json={"q": "hello"})

    assert response.status_code == 429
    headers = {key.lower(): value for key, value in response.headers.items()}
    assert "retry-after" in headers

    monkeypatch.delenv("RATE_LIMIT_QPS", raising=False)
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    importlib.reload(config)
    importlib.reload(api_main)


def test_cors_headers_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    def _configure_env() -> None:
        monkeypatch.delenv("RATE_LIMIT_QPS", raising=False)
        monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://app.example,https://web.example")

    config, api_main = _reload_app(monkeypatch, _configure_env)

    client = TestClient(api_main.app)
    headers = {
        "Origin": "https://app.example",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/upload", headers=headers)

    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == "https://app.example"

    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    importlib.reload(config)
    importlib.reload(api_main)
