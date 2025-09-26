from __future__ import annotations

from typing import TYPE_CHECKING

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


settings = Settings()
