"""FastAPI application exposing simple upload, index, and query endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from fastapi import FastAPI, File, HTTPException, UploadFile, status

try:
    from . import local_llm  # type: ignore
except Exception:  # pragma: no cover - fallback when optional dependency missing

    class _DefaultLocalLLM:
        """Fallback implementation used when a custom local LLM is unavailable."""

        @staticmethod
        def generate(prompt: str) -> str:
            return f"Local response: {prompt}"

    local_llm = _DefaultLocalLLM()  # type: ignore


def _resolve_books_dir() -> Path:
    """Return the directory used to persist uploaded books."""
    env_path = os.environ.get("KETABMIND_BOOK_DIR")
    base_path = Path(env_path) if env_path else Path.home() / ".ketabmind" / "books"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


BOOKS_DIR: Path = _resolve_books_dir()
INDEXED_FILES: set[str] = set()
UPLOAD_FILE_PARAM = File(...)


class UploadResponse(BaseModel):
    filename: str = Field(..., description="Stored filename")
    path: Path = Field(..., description="Absolute path of the stored file")


class IndexRequest(BaseModel):
    filename: str = Field(..., description="Filename previously uploaded via /upload")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata payload")


class IndexResponse(BaseModel):
    filename: str
    indexed: bool


class QueryRequest(BaseModel):
    prompt: str = Field(..., description="Prompt forwarded to the local LLM")


class QueryResponse(BaseModel):
    response: str


app = FastAPI(title="KetabMind Local API")


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health-check endpoint."""
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    """Return the backend version string."""
    return {"version": "0.1.0"}


@app.post("/upload")
async def upload(file: UploadFile = UPLOAD_FILE_PARAM) -> dict[str, Any]:
    """Persist the uploaded file under the configured books directory."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename")

    safe_name = Path(file.filename).name
    target_path = BOOKS_DIR / safe_name

    contents = await file.read()
    target_path.write_bytes(contents)

    response = UploadResponse(filename=safe_name, path=target_path)
    return response.model_dump()


@app.post("/index")
def index(request: IndexRequest) -> dict[str, Any]:
    """Mark the provided filename as indexed if it exists on disk."""
    target_path = BOOKS_DIR / Path(request.filename).name
    if not target_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    INDEXED_FILES.add(target_path.name)
    response = IndexResponse(filename=target_path.name, indexed=True)
    return response.model_dump()


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """Proxy the prompt to the local LLM implementation and return its response."""
    generate = getattr(local_llm, "generate", None)
    if not callable(generate):  # pragma: no cover - defensive branch
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM backend unavailable",
        )

    response_text = generate(request.prompt)
    response = QueryResponse(response=response_text)
    return response.model_dump()


__all__ = [
    "app",
    "health",
    "index",
    "query",
    "upload",
    "version",
]
