from __future__ import annotations

import importlib
import sys
import time
import types

import pytest

from fastapi.testclient import TestClient


@pytest.fixture
def auth_app(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    if "limits" not in sys.modules:
        limits_stub = types.ModuleType("limits")

        class RateLimitItemPerSecond:  # type: ignore[too-few-public-methods]
            def __init__(self, amount: int, multiples: int) -> None:
                self.amount = amount
                self.multiples = multiples

        limits_stub.RateLimitItemPerSecond = RateLimitItemPerSecond
        monkeypatch.setitem(sys.modules, "limits", limits_stub)

    if "slowapi" not in sys.modules:
        slowapi_stub = types.ModuleType("slowapi")

        class _LimiterBackend:
            def test(self, *args, **kwargs):
                return True

            def hit(self, *args, **kwargs) -> None:
                return None

            def get_window_stats(self, *args, **kwargs):
                return (time.time(), 0)

        class Limiter:  # type: ignore[too-few-public-methods]
            def __init__(self, key_func=None, *, enabled: bool = False, **kwargs) -> None:
                self._key_func = key_func
                self.enabled = enabled
                self.limiter = _LimiterBackend()

        slowapi_stub.Limiter = Limiter
        monkeypatch.setitem(sys.modules, "slowapi", slowapi_stub)
    if "prometheus_client" not in sys.modules:
        prom_stub = types.ModuleType("prometheus_client")

        class _Metric:  # type: ignore[too-few-public-methods]
            def __init__(self, *args, **kwargs) -> None:
                return None

            def labels(self, *args, **kwargs):
                return self

            def inc(self, *args, **kwargs) -> None:
                return None

            def observe(self, *args, **kwargs) -> None:
                return None

            def set(self, *args, **kwargs) -> None:
                return None

        prom_stub.CONTENT_TYPE_LATEST = "text/plain"
        prom_stub.Counter = _Metric
        prom_stub.Gauge = _Metric
        prom_stub.Histogram = _Metric

        def generate_latest() -> bytes:
            return b""

        prom_stub.generate_latest = generate_latest
        monkeypatch.setitem(sys.modules, "prometheus_client", prom_stub)
    import core.config as config

    importlib.reload(config)
    config.reload_settings()

    import apps.api.main as api_main

    importlib.reload(api_main)
    yield api_main.app

    monkeypatch.delenv("AUTH_REQUIRED", raising=False)
    config.reload_settings()
    importlib.reload(api_main)


def test_login_sets_cookie(auth_app) -> None:
    client = TestClient(auth_app)
    response = client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "wonderland"},
    )
    assert response.status_code == 200
    set_cookie = response.headers.get("set-cookie", "")
    assert "access_token=" in set_cookie
    data = response.json()
    assert data["user"]["email"] == "alice@example.com"
    assert response.headers.get("x-csrf-token")


def test_protected_route_without_cookie(auth_app) -> None:
    client = TestClient(auth_app)
    response = client.post("/index", json={"path": "/tmp/missing.txt"})
    assert response.status_code == 401


def test_user_cannot_access_other_resources(auth_app) -> None:
    client = TestClient(auth_app)
    login_response = client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "wonderland"},
    )
    assert login_response.status_code == 200
    forbidden_response = client.get("/bookmarks", params={"user_id": "user-bob"})
    assert forbidden_response.status_code == 403
