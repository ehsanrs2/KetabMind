from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", case_sensitive=False)

    qdrant_mode: str = "local"  # local | remote
    qdrant_location: str = "./qdrant_local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "books"

    embed_model: str = "mock"  # mock|small|base

    chunk_size: int = 800
    chunk_overlap: int = 200


settings = Settings()
