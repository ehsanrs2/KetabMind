from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_origin, get_type_hints

if TYPE_CHECKING:

    from pydantic_settings import BaseSettings as BaseSettingsProto
    from pydantic_settings import SettingsConfigDict

else:
    _pydantic_settings_spec = importlib.util.find_spec("pydantic_settings")

    if _pydantic_settings_spec is not None:  # pragma: no cover - import side effect
        _pydantic_settings = importlib.import_module("pydantic_settings")
        SettingsConfigDict = _pydantic_settings.SettingsConfigDict
        BaseSettingsProto = _pydantic_settings.BaseSettings
    else:
        class SettingsConfigDict(dict):
            """Lightweight mapping compatible with pydantic's SettingsConfigDict."""

            env_file: str | None

            def __init__(self, env_file: str | None = None, **kwargs: Any) -> None:
                super().__init__(env_file=env_file, **kwargs)
                self.env_file = env_file

        class BaseSettingsProto:  # pragma: no cover - simple fallback implementation
            """Fallback replacement for pydantic BaseSettings.

            It supports env-file loading and basic type coercion for common scalar
            types used in the settings model.
            """

            model_config: SettingsConfigDict
            model_fields: dict[str, Any]

            def __init_subclass__(cls, **kwargs: Any) -> None:
                super().__init_subclass__(**kwargs)
                annotations = get_type_hints(cls, include_extras=True)
                cls.model_fields = {name: None for name in annotations}

            def __init__(self, **overrides: Any) -> None:
                annotations = get_type_hints(self.__class__, include_extras=True)
                self.model_fields = {name: None for name in annotations}
                env: dict[str, str] = {}
                env_file = getattr(self.model_config, "env_file", None)
                if env_file:
                    env_path = Path(env_file)
                    if env_path.is_file():
                        env.update(_read_env_file(env_path))

                for name in self.model_fields:
                    key = name.upper()
                    raw_value = overrides.get(name)
                    if raw_value is None:
                        raw_value = os.getenv(key, env.get(key))

                    if raw_value is None:
                        value = getattr(self.__class__, name)
                    else:
                        annotation = annotations.get(name, str)
                        value = _coerce(raw_value, annotation)

                    setattr(self, name, value)

            def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                return {name: getattr(self, name) for name in self.model_fields}


def _read_env_file(path: Path) -> dict[str, str]:
    env_data: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env_data[key.strip()] = value.strip().strip('"').strip("'")
    return env_data


def _coerce(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        target = annotation
    elif origin is list:
        (inner,) = get_args(annotation) or (str,)
        return [_coerce(item, inner) for item in value.split(",")] if isinstance(value, str) else value
    elif origin is tuple:
        inner = get_args(annotation)
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            coerced = [_coerce(part, inner[idx] if idx < len(inner) else str) for idx, part in enumerate(parts)]
            return tuple(coerced)
        return tuple(value)
    else:
        target = get_args(annotation)[0]

    if target in (Any, None, type(None)):
        return value
    if target is bool:
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        return lowered in {"1", "true", "yes", "on"}
    if target is int:
        return int(value)
    if target is float:
        return float(value)
    if target is str:
        return str(value)
    return value


class Settings(BaseSettingsProto):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", case_sensitive=False)

    qdrant_mode: str = "local"  # local | remote
    qdrant_location: str = "./qdrant_local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "books"

    embed_model: str = "mock"  # mock|small|base

    reranker_enabled: bool = False
    reranker_model_name: str = "bge-reranker-v2-m3"
    reranker_topk: int = 50
    reranker_batch: int = 16
    reranker_cache_size: int = 10000

    hybrid_weights: str = "cosine=0.4,lexical=0.2,reranker=0.4"
    lexical_fa_preproc: bool = True

    llm_backend: str = "mock"
    llm_model: str = "mock"
    llm_max_input_tokens: int = 4096
    llm_max_new_tokens: int = 256
    llm_temperature: float = 0.2
    llm_top_p: float = 0.95
    ollama_host: str = "http://localhost:11434"

    chunk_size: int = 800
    chunk_overlap: int = 200

    ingest_header_lines: int = 0
    ingest_footer_lines: int = 0

    rate_limit_qps: float | None = None
    cors_allow_origins: list[str] = ["*"]

    auth_required: bool = False
    jwt_secret: str = "change-me"
    jwt_expiration_seconds: int = 3600


_CACHE_KEYS = tuple(name.upper() for name in Settings.model_fields)
_cached_settings: Settings | None = None
_cached_signature: tuple[tuple[str, str | None], ...] | None = None


def _build_signature() -> tuple[tuple[str, str | None], ...]:
    return tuple((key, os.getenv(key)) for key in _CACHE_KEYS)


def _load_settings(force: bool = False) -> Settings:
    global _cached_settings, _cached_signature
    signature = _build_signature()
    if force or _cached_settings is None or signature != _cached_signature:
        _cached_settings = Settings()
        _cached_signature = signature
    return _cached_settings


def get_settings(*, reload: bool = False) -> Settings:
    """Return current settings, reloading when environment changes."""

    return _load_settings(force=reload)


def reload_settings() -> Settings:
    """Force settings reload from environment."""

    return _load_settings(force=True)


class _SettingsProxy:
    """Lightweight proxy exposing the current settings instance."""

    def __getattr__(self, item: str) -> Any:
        return getattr(get_settings(), item)

    def __setattr__(self, key: str, value: Any) -> None:
        setattr(get_settings(), key, value)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return repr(get_settings())

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return get_settings().model_dump(*args, **kwargs)


settings = _SettingsProxy()

__all__ = ["Settings", "settings", "get_settings", "reload_settings"]
