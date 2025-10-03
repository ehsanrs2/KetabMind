"""Tests for the lightweight settings fallback in :mod:`core.config`."""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest


def _force_missing_pydantic_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure importing ``pydantic_settings`` raises ``ImportError``."""

    original_import = builtins.__import__

    def _raise_import(name: str, *args: Any, **kwargs: Any):
        if name == "pydantic_settings" or name.startswith("pydantic_settings."):
            raise ImportError("forced missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pydantic_settings", raising=False)
    monkeypatch.setattr(builtins, "__import__", _raise_import)


def _reload_config(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Reload ``core.config`` under forced import conditions."""

    import core.config as config

    reloaded = importlib.reload(config)
    monkeypatch.setattr(reloaded, "_cached_settings", None)
    monkeypatch.setattr(reloaded, "_cached_signature", None)
    return reloaded


def test_settings_fallback_env_and_sequence_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fallback ``Settings`` should read environment variables and coerce sequences."""

    _force_missing_pydantic_settings(monkeypatch)
    reloaded = _reload_config(monkeypatch)

    env_file = tmp_path / ".env"
    expected_value = "value-from-env"
    env_file.write_text(
        f"APP_JWT_SECRET={expected_value}\nAPP_UPLOAD_SIGNED_URL_TTL=180\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("APP_RERANKER_ENABLED", "yes")
    monkeypatch.setenv("APP_CORS_ALLOW_ORIGINS", "https://one.test, https://two.test")

    monkeypatch.setattr(
        reloaded.Settings,
        "model_config",
        reloaded.SettingsConfigDict(
            env_file=str(env_file), env_prefix="APP_", case_sensitive=False
        ),
    )

    settings = reloaded.reload_settings()

    assert settings.reranker_enabled is True
    assert settings.cors_allow_origins == ["https://one.test", "https://two.test"]
    assert settings.jwt_secret == expected_value
    assert settings.upload_signed_url_ttl == 180
    assert reloaded._coerce("1,2", tuple[int, int]) == (1, 2)
    assert reloaded._coerce(["a", "b"], list[str]) == ["a", "b"]


def test_settings_fallback_respects_case_sensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """When case sensitivity is enabled, exact key casing should be required."""

    _force_missing_pydantic_settings(monkeypatch)
    reloaded = _reload_config(monkeypatch)

    monkeypatch.setattr(
        reloaded.Settings,
        "model_config",
        reloaded.SettingsConfigDict(env_prefix="app_", case_sensitive=True),
    )

    monkeypatch.setenv("app_reranker_enabled", "true")
    settings = reloaded.reload_settings()

    # Case sensitive prefix requires the lowercase env key to match exactly.
    assert settings.reranker_enabled is True
