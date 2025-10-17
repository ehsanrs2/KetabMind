"""FastAPI application exposing simple upload, index, and query endpoints."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

try:
    from . import local_llm  # type: ignore
except Exception:  # pragma: no cover - fallback when optional dependency missing

    class _DefaultLocalLLM:
        """Fallback implementation used when a custom local LLM is unavailable."""

        @staticmethod
        def generate(prompt: str) -> str:
            return f"Local response: {prompt}"

        @staticmethod
        async def generate_stream(prompt: str, model: str | None = None) -> AsyncIterator[str]:
            """Yield a simulated streaming response for development use."""

            words = prompt.split()
            for index, word in enumerate(words):
                suffix = "" if index == len(words) - 1 else " "
                await asyncio.sleep(0.05)
                yield f"{word}{suffix}"

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


class StreamMessageRequest(BaseModel):
    """Request payload for streaming assistant responses."""

    content: str = Field(..., description="User prompt forwarded to the model")
    model: Literal["ollama", "openai"] = Field(
        ..., description="Identifier of the target language model"
    )
    temperature: float | None = Field(
        default=None, description="Optional sampling temperature hint"
    )
    context: bool | None = Field(
        default=None, description="Whether to include prior messages in the prompt"
    )


app = FastAPI(title="KetabMind Local API")
SESSIONS: list[Session] = []
messages_store: dict[str, list[Message]] = {}


def _get_session(session_id: UUID) -> Session | None:
    """Return the session matching the provided identifier if it exists."""

    for session in SESSIONS:
        if session.id == session_id:
            return session
    return None


def _append_message(session: Session, role: Literal["user", "assistant"], content: str) -> Message:
    """Persist a message for the given session in the in-memory store."""

    message = Message(
        id=str(uuid4()),
        session_id=str(session.id),
        role=role,
        content=content,
        created_at=datetime.utcnow(),
    )
    messages = messages_store.setdefault(str(session.id), [])
    messages.append(message)
    return message


def _build_prompt_for_model(
    session: Session,
    user_message: Message,
    include_context: bool,
) -> str:
    """Construct a prompt string combining history and the latest user message."""

    if include_context:
        history: list[Message] = []
        for message in messages_store.get(str(session.id), []):
            history.append(message)
            if message.id == user_message.id:
                break
    else:
        history = [user_message]

    lines = [f"{message.role}: {message.content}" for message in history]
    lines.append("assistant:")
    return "\n".join(lines)


async def _simulate_streaming_response(text: str) -> AsyncIterator[str]:
    """Yield text chunks with a small delay to mimic streaming tokens."""

    words = text.split()
    if not words:
        return

    for index, word in enumerate(words):
        suffix = "" if index == len(words) - 1 else " "
        await asyncio.sleep(0.05)
        yield f"{word}{suffix}"


def _compose_simulated_response(
    session: Session,
    payload: StreamMessageRequest,
) -> str:
    """Generate a deterministic simulated answer for development purposes."""

    history = messages_store.get(str(session.id), [])
    history_count = max(len(history) - 1, 0)
    model_label = "مدل محلی" if payload.model == "ollama" else "مدل ابری"
    context_clause = ""
    if payload.context and history_count:
        context_clause = f" پس از مرور {history_count} پیام قبلی"
    elif payload.context:
        context_clause = " با استفاده از گفت‌وگوی فعلی"
    temperature_clause = ""
    if payload.temperature is not None:
        temperature_clause = f" مقدار دما {payload.temperature:.1f} تنظیم شده است."
    user_prompt = payload.content.strip() or "پرسش"
    return (
        f"{model_label} KetabMind{context_clause} جمع‌بندی می‌کند که برای پرسش «{user_prompt}» "
        f"مطالعه بخش‌های کلیدی کتاب و یادداشت‌برداری دقیق سودمند است.{temperature_clause}"
    )


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

    return _append_message(session, payload.role, payload.content)


@app.post("/sessions/{session_id}/messages/stream")
async def stream_session_message(
    session_id: UUID, payload: StreamMessageRequest
) -> StreamingResponse:
    """Stream an assistant response for the provided session using server-sent events."""

    session = _get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    user_message = _append_message(session, "user", payload.content)
    include_context = bool(payload.context)
    prompt = _build_prompt_for_model(session, user_message, include_context)

    async def _model_stream() -> AsyncIterator[str]:
        if payload.model == "ollama":
            stream_fn = getattr(local_llm, "generate_stream", None)
            if callable(stream_fn):
                async for chunk in stream_fn(prompt, model=payload.model):
                    yield chunk
                return
        response_text = _compose_simulated_response(session, payload)
        async for chunk in _simulate_streaming_response(response_text):
            yield chunk

    async def _event_stream() -> AsyncIterator[str]:
        collected: list[str] = []
        async for piece in _model_stream():
            collected.append(piece)
            yield f"data: {piece}\n\n"
        final_text = "".join(collected).rstrip()
        _append_message(session, "assistant", final_text)
        yield "data: [DONE]\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


__all__ = [
    "app",
    "create_session_message",
    "create_session",
    "delete_session",
    "health",
    "index",
    "list_session_messages",
    "list_sessions",
    "stream_session_message",
    "query",
    "upload",
    "version",
]
