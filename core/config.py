from __future__ import annotations

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic_settings import SettingsConfigDict

if TYPE_CHECKING:

    class BaseSettingsProto:
        model_config: SettingsConfigDict

else:  # pragma: no cover - runtime import
    from pydantic_settings import BaseSettings as BaseSettingsProto


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
