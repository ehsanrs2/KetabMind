# mypy: ignore-errors
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str | None = None
    qdrant_mode: str = "remote"
    qdrant_collection: str = "ketabmind"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
