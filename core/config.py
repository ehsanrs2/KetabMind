# mypy: ignore-errors
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_url: str | None = None
    qdrant_mode: str = "remote"
    qdrant_location: str = "./qdrant_local"
    qdrant_collection: str = "ketabmind"
    ingest_header_lines: int = 0
    ingest_footer_lines: int = 0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
