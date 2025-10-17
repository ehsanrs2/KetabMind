"""FastAPI application exposing simple upload, index, and query endpoints."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

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


class Session(BaseModel):
    """Payload representing a single chat session."""

    id: UUID = Field(..., description="Unique identifier for the session")
    title: str = Field(..., description="Human-friendly session title")
    created_at: datetime = Field(..., description="UTC timestamp marking session creation")


class SessionCreate(BaseModel):
    """Request payload for creating a new chat session."""

    title: str = Field(..., description="Title assigned to the new session")


class Message(BaseModel):
    """Representation of a single message exchanged within a chat session."""

    id: str = Field(..., description="Unique identifier for the message")
    session_id: str = Field(..., description="Identifier of the session owning the message")
    role: Literal["user", "assistant"] = Field(
        ..., description="Source of the message within the conversation"
    )
    content: str = Field(..., description="Message payload")
    created_at: datetime = Field(..., description="UTC timestamp when the message was created")


class MessageCreate(BaseModel):
    """Payload required to create a new chat message."""

    role: Literal["user", "assistant"] = Field(
        ..., description="Source of the message within the conversation"
    )
    content: str = Field(..., description="Message payload")


app = FastAPI(title="KetabMind Local API")
SESSIONS: list[Session] = []
messages_store: dict[str, list[Message]] = {}


def _get_session(session_id: UUID) -> Session | None:
    """Return the session matching the provided identifier if it exists."""

    for session in SESSIONS:
        if session.id == session_id:
            return session
    return None


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


@app.get("/sessions", response_model=list[Session])
def list_sessions() -> list[Session]:
    """Return all stored chat sessions."""

    return SESSIONS


@app.post("/sessions", response_model=Session, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreate) -> Session:
    """Create and persist an in-memory chat session."""

    session = Session(id=uuid4(), title=payload.title, created_at=datetime.utcnow())
    SESSIONS.append(session)
    return session


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: UUID) -> None:
    """Remove the chat session matching the provided identifier."""

    session = _get_session(session_id)
    if session is None:  # pragma: no cover - simple branch
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    SESSIONS.remove(session)
    messages_store.pop(str(session.id), None)


@app.get("/sessions/{session_id}/messages", response_model=list[Message])
def list_session_messages(session_id: UUID) -> list[Message]:
    """Return the ordered list of messages associated with a session."""

    session = _get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    return messages_store.get(str(session.id), [])


@app.post(
    "/sessions/{session_id}/messages",
    response_model=Message,
    status_code=status.HTTP_201_CREATED,
)
def create_session_message(session_id: UUID, payload: MessageCreate) -> Message:
    """Create and persist a new message within the specified session."""

    session = _get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    message = Message(
        id=str(uuid4()),
        session_id=str(session.id),
        role=payload.role,
        content=payload.content,
        created_at=datetime.utcnow(),
    )
    messages = messages_store.setdefault(str(session.id), [])
    messages.append(message)
    return message


__all__ = [
    "app",
    "create_session_message",
    "create_session",
    "delete_session",
    "health",
    "index",
    "list_session_messages",
    "list_sessions",
    "query",
    "upload",
    "version",
]
