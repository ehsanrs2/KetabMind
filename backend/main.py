"""FastAPI application exposing simple upload, index, and query endpoints."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse

from apps.api.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    get_user_bookmarks,
    get_user_profile,
)
from core.config import settings
from core.answer.answerer import stream_answer

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


class LoginRequest(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")


def _csrf_headers(token: str | None) -> dict[str, str]:
    """Return CSRF response headers when a token is available."""

    if token:
        return {"x-csrf-token": token}
    return {}


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


@app.post("/auth/login")
def login(request: LoginRequest) -> JSONResponse:
    """Authenticate the user and issue a cookie-based access token."""

    record = authenticate_user(request.email, request.password)
    token = create_access_token({"sub": record["id"], "email": record["email"]})
    profile = get_user_profile(record["id"])
    max_age = settings.jwt_expiration_seconds
    cookie_parts = [f"access_token={token}", "HttpOnly", "Path=/", "SameSite=Lax"]
    if max_age:
        cookie_parts.append(f"Max-Age={int(max_age)}")
    headers = {"set-cookie": "; ".join(cookie_parts), **_csrf_headers(token)}
    return JSONResponse({"user": profile}, headers=headers)


@app.get("/me")
def me(
    request: Request,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> JSONResponse:
    """Return the current user profile along with a CSRF token header."""

    cookie_token = request.cookies.get("access_token") if hasattr(request, "cookies") else None
    token = cookie_token if isinstance(cookie_token, str) and cookie_token else None
    return JSONResponse(current_user, headers=_csrf_headers(token))


@app.post("/auth/logout")
def logout() -> JSONResponse:
    """Clear the auth cookie and return a success status."""

    response = JSONResponse({"status": "ok"}, headers=_csrf_headers(None))
    response.delete_cookie("access_token", path="/")
    return response


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


@app.get("/bookmarks")
def list_bookmarks(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)]
) -> list[dict[str, Any]]:
    """Return the bookmark collection for the current user."""

    user_id = current_user.get("id")
    if not isinstance(user_id, str):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid user profile")
    return get_user_bookmarks(user_id)


@app.post("/sessions", response_model=Session, status_code=status.HTTP_201_CREATED)
def create_session(payload: SessionCreate) -> Session:
    """Create and persist an in-memory chat session."""

    session = Session(id=uuid4(), title=payload.title, created_at=datetime.utcnow())
    SESSIONS.append(session)
    return session


@app.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_session(session_id: UUID) -> Response:
    """Remove the chat session matching the provided identifier."""

    session = _get_session(session_id)
    if session is None:  # pragma: no cover - simple branch
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    SESSIONS.remove(session)
    messages_store.pop(str(session.id), None)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
    session_id: UUID,
    payload: StreamMessageRequest,
    book_id: Annotated[str | None, Query()] = None,
) -> StreamingResponse:
    """Stream an assistant response for the provided session using server-sent events."""

    session = _get_session(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    question = (payload.content or "").strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User message is empty")

    _append_message(session, "user", question)

    async def _event_stream() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        final_answer: str | None = None
        final_contexts: list[dict[str, Any]] = []
        final_meta: dict[str, Any] | None = None
        error_message: str | None = None

        def _produce() -> None:
            try:
                for chunk in stream_answer(question, book_id=book_id):
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
            except Exception as exc:  # pragma: no cover - defensive
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            else:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        producer_task = asyncio.create_task(asyncio.to_thread(_produce))

        try:
            while True:
                kind, item = await queue.get()
                if kind == "chunk":
                    chunk = cast(dict[str, Any], item)
                    delta = chunk.get("delta")
                    if isinstance(delta, str) and delta:
                        yield "data: " + json.dumps({"delta": delta}, ensure_ascii=False) + "\n\n"
                        continue
                    if "answer" not in chunk:
                        continue
                    final_answer = str(chunk.get("answer") or "")
                    raw_contexts = chunk.get("contexts") or []
                    final_contexts = [dict(ctx) for ctx in raw_contexts if isinstance(ctx, Mapping)]
                    raw_meta = chunk.get("meta")
                    final_meta = raw_meta if isinstance(raw_meta, Mapping) else None
                    data: dict[str, Any] = {"answer": final_answer, "contexts": final_contexts}
                    if final_meta:
                        data["meta"] = final_meta
                    debug_payload = chunk.get("debug")
                    if debug_payload is not None:
                        data["debug"] = debug_payload
                    yield "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
                elif kind == "error":
                    error_message = str(item)
                    error_payload = {"error": error_message}
                    yield "data: " + json.dumps(error_payload, ensure_ascii=False) + "\n\n"
                    break
                elif kind == "done":
                    break
        except asyncio.CancelledError:  # pragma: no cover - client cancelled
            error_message = "cancelled"
        finally:
            await producer_task

        if error_message is None and final_answer is not None:
            _append_message(session, "assistant", final_answer)

        yield "data: [DONE]\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


__all__ = [
    "app",
    "list_bookmarks",
    "login",
    "logout",
    "me",
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
