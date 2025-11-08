from __future__ import annotations

import asyncio
import base64
import binascii
import gzip
import hashlib
import hmac
import inspect
import json
import math
import mimetypes
import re
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, cast

from answering.citations import build_citations
from limits import RateLimitItemPerSecond
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from slowapi import Limiter
from sqlalchemy import func, select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

import structlog
from apps.api.auth import authenticate_user, create_access_token, get_current_user, get_user_profile
from apps.api.db import models, repositories
from apps.api.db.session import session_scope
from apps.api.middleware import RequestIDMiddleware
from apps.api.routes.query import build_query_response
from apps.api.schemas import (
    BookmarkCreate,
    ExportRequest,
    IndexRequest,
    MessageCreate,
    Metadata,
    SessionCreate,
    SessionUpdate,
    StreamMessageRequest,
)
from core.answer.answerer import answer, stream_answer
from core.answer.llm import LLMError, LLMTimeoutError, get_llm
from core.config import settings
from core.fts import FTSMatch
from core.fts import get_backend as get_fts_backend
from exporting import export_answer
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from utils.hash import sha256_file
from utils.logging import configure_logging
from utils.pydantic_compat import BaseModel

if TYPE_CHECKING:
    from core.index import IndexedFile as IndexedFileType
    from core.index import IndexResult as IndexResultType

    class BaseModelProto:
        pass

else:  # pragma: no cover - runtime import
    IndexResultType = Any
    IndexedFileType = Any
    BaseModelProto = BaseModel

configure_logging()

log = structlog.get_logger(__name__)

def _client_identifier(request: Request) -> str:
    headers = cast(Mapping[str, str], getattr(request, "headers", {}))
    forwarded = headers.get("x-forwarded-for")
    if isinstance(forwarded, str):
        return forwarded.split(",", 1)[0].strip()
    real_ip = headers.get("x-real-ip")
    if isinstance(real_ip, str):
        return real_ip
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client else None
    if isinstance(host, str):
        return host
    return "127.0.0.1"


class _GZipMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, minimum_size: int = 500) -> None:
        super().__init__(app)
        self.minimum_size = minimum_size

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        response = await call_next(request)
        body = getattr(response, "_content", None)
        if body is None:
            return response
        existing_encoding = response.headers.get("content-encoding")
        if existing_encoding:
            return response
        if isinstance(body, bytes):
            raw = body
        elif isinstance(body, str):
            raw = body.encode("utf-8")
        else:
            try:
                raw = json.dumps(body).encode("utf-8")
            except TypeError:
                return response
        if len(raw) < self.minimum_size:
            return response
        compressed = gzip.compress(raw)
        response.headers["content-encoding"] = "gzip"
        response.headers.setdefault("vary", "Accept-Encoding")
        response.headers["content-length"] = str(len(compressed))
        response._content = compressed
        return response


app = FastAPI(title="KetabMind API")

SESSION_CLEANUP_INTERVAL_SECONDS = 3600
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")


def _cleanup_expired_sessions_once() -> int:
    retention_days = settings.history_retention_days
    if retention_days <= 0:
        return 0
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    with session_scope() as db_session:
        removed = repositories.delete_sessions_older_than(db_session, older_than=cutoff)
        if removed:
            log.info(
                "sessions.cleanup",
                removed=removed,
                cutoff=cutoff.isoformat(),
            )
        return removed


async def _session_cleanup_worker() -> None:
    await asyncio.sleep(5)
    while True:
        try:
            _cleanup_expired_sessions_once()
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            raise
        except Exception:  # pragma: no cover - defensive logging
            log.warning("sessions.cleanup_failed", exc_info=True)
        await asyncio.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)


rate_limit_qps = settings.rate_limit_qps
limiter_enabled = bool(rate_limit_qps and rate_limit_qps > 0)
default_limits: list[str] = []
if limiter_enabled:
    per_second = max(1, math.ceil(rate_limit_qps))
    default_limits.append(f"{per_second}/second")

limiter = Limiter(
    key_func=lambda request: _client_identifier(request),
    default_limits=default_limits,
    headers_enabled=True,
    enabled=limiter_enabled,
)
app.state.limiter = limiter

_GLOBAL_RATE_LIMIT: RateLimitItemPerSecond | None = None
_QUERY_RATE_LIMIT: RateLimitItemPerSecond | None = None
if limiter.enabled and rate_limit_qps:
    base_limit = max(1, math.ceil(rate_limit_qps))
    _GLOBAL_RATE_LIMIT = RateLimitItemPerSecond(base_limit, 1)
    query_per_second = max(1, math.ceil(base_limit / 2))
    if query_per_second >= base_limit and base_limit > 1:
        query_per_second = base_limit - 1
    _QUERY_RATE_LIMIT = RateLimitItemPerSecond(max(1, query_per_second), 1)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(_GZipMiddleware, minimum_size=500)

cors_allow_origins = settings.cors_allow_origins or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _on_startup() -> None:
    if settings.history_retention_days <= 0:
        return
    task = asyncio.create_task(_session_cleanup_worker())
    app.state.session_cleanup_task = task


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    task = getattr(app.state, "session_cleanup_task", None)
    if task is None:
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


if "REQUEST_COUNTER" not in globals():
    existing_counter = REGISTRY._names_to_collectors.get("api_requests_total")
    if isinstance(existing_counter, Counter):
        REQUEST_COUNTER = existing_counter
    else:
        REQUEST_COUNTER = Counter(
            "api_requests_total",
            "Total number of API requests",
            labelnames=("path", "method", "status"),
        )
if "REQUEST_LATENCY" not in globals():
    existing_latency = REGISTRY._names_to_collectors.get("api_request_latency_seconds")
    if isinstance(existing_latency, Histogram):
        REQUEST_LATENCY = existing_latency
    else:
        REQUEST_LATENCY = Histogram(
            "api_request_latency_seconds",
            "Latency of API requests in seconds",
            labelnames=("path", "method"),
        )
if "ERROR_COUNTER" not in globals():
    existing_error = REGISTRY._names_to_collectors.get("api_errors_total")
    if isinstance(existing_error, Counter):
        ERROR_COUNTER = existing_error
    else:
        ERROR_COUNTER = Counter(
            "api_errors_total",
            "Total number of API errors",
            labelnames=("path", "method"),
        )
if "TOKEN_USAGE_GAUGE" not in globals():
    existing_token = REGISTRY._names_to_collectors.get("token_usage")
    if isinstance(existing_token, Gauge):
        TOKEN_USAGE_GAUGE = existing_token
    else:
        TOKEN_USAGE_GAUGE = Gauge(
            "token_usage",
            "Token usage reported by the Answer Agent",
        )

F = TypeVar("F", bound=Callable[..., Any])


def _rate_limit_response(retry_after: int) -> JSONResponse:
    response = JSONResponse(
        {"error": {"type": "rate_limit", "message": "Too many requests"}},
        status_code=429,
    )
    response.headers["Retry-After"] = str(max(1, retry_after))
    return response


def _enforce_limit(limit: RateLimitItemPerSecond | None, request: Request) -> int | None:
    if not limiter.enabled or limit is None:
        return None
    key_func = getattr(limiter, "_key_func", None)
    key = key_func(request) if callable(key_func) else _client_identifier(request)
    if limiter.limiter.test(limit, key):
        limiter.limiter.hit(limit, key)
        return None
    reset_time, _remaining = limiter.limiter.get_window_stats(limit, key)
    retry_after = max(1, int(math.ceil(reset_time - time.time())))
    return retry_after


def _get(path: str) -> Callable[[F], F]:
    return cast(Callable[[F], F], app.get(path))


def _post(path: str) -> Callable[[F], F]:
    return cast(Callable[[F], F], app.post(path))


def _delete(path: str) -> Callable[[F], F]:
    delete_handler = getattr(app, "delete", None)
    if callable(delete_handler):
        return cast(Callable[[F], F], delete_handler(path))

    def decorator(func: F) -> F:
        add_route = getattr(app, "_add_route", None)
        if callable(add_route):
            add_route("DELETE", path, func)
            return func
        raise RuntimeError("DELETE method not supported by FastAPI stub")

    return decorator


def _patch(path: str) -> Callable[[F], F]:
    patch_handler = getattr(app, "patch", None)
    if callable(patch_handler):
        return cast(Callable[[F], F], patch_handler(path))

    def decorator(func: F) -> F:
        add_route = getattr(app, "_add_route", None)
        if callable(add_route):
            add_route("PATCH", path, func)
            return func
        raise RuntimeError("PATCH method not supported by FastAPI stub")

    return decorator


def _parse_identifier(value: Any, *, field: str) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            with suppress(ValueError):
                return int(stripped)
    raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Invalid {field}")


def _isoformat(timestamp: datetime | None) -> str | None:
    if timestamp is None:
        return None
    return timestamp.isoformat()


def _safe_json_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    cleaned: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            cleaned.append(item)
    return cleaned or None


def _safe_json_mapping(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return None

def _persist_assistant_message(
    session_id: int,
    owner_id: int,
    content: str,
    *,
    citations: Sequence[str] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Persist the assistant response for the provided session."""

    if not content:
        return
    stored_citations = list(citations) if citations else None
    stored_meta = dict(meta) if isinstance(meta, Mapping) else None
    with session_scope() as db_session:
        message_repo = repositories.MessageRepository(db_session, owner_id)
        try:
            message_repo.create(
                session_id=session_id,
                role="assistant",
                content=content,
                citations=stored_citations,
                meta=stored_meta,
            )
        except ValueError:  # pragma: no cover - session removed concurrently
            log.warning(
                "stream.persist_failed",
                session_id=session_id,
                owner_id=owner_id,
            )


def _detect_message_language(text: str) -> str:
    if text and _ARABIC_SCRIPT_RE.search(text):
        return "fa"
    return "en"


def _average_context_confidence(contexts: Sequence[Mapping[str, Any]]) -> float:
    scores: list[float] = []
    for ctx in contexts:
        if not isinstance(ctx, Mapping):
            continue
        value = ctx.get("hybrid")
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _build_message_citations(contexts: Sequence[Mapping[str, Any]]) -> list[str]:
    normalized: list[dict[str, Any]] = []
    for ctx in contexts:
        if isinstance(ctx, Mapping):
            normalized.append({**ctx})
    if not normalized:
        return []
    return build_citations(normalized, "[${book_id}:${page_range}]")


def _derive_message_meta(
    answer: str,
    contexts: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if isinstance(meta, Mapping):
        payload.update(meta)

    lang = payload.get("lang")
    if not isinstance(lang, str) or not lang:
        payload["lang"] = _detect_message_language(answer)

    coverage_value = payload.get("coverage")
    if coverage_value is None:
        payload["coverage"] = 0.0
    else:
        try:
            payload["coverage"] = float(coverage_value)
        except (TypeError, ValueError):
            payload["coverage"] = 0.0

    confidence_value = payload.get("confidence")
    if confidence_value is None:
        payload["confidence"] = _average_context_confidence(contexts)
    else:
        try:
            payload["confidence"] = float(confidence_value)
        except (TypeError, ValueError):
            payload["confidence"] = _average_context_confidence(contexts)

    return payload


def _serialize_message(message: models.Message) -> dict[str, Any]:
    citations = _safe_json_list(message.citations)
    meta = _safe_json_mapping(message.meta)
    return {
        "id": message.id,
        "session_id": message.session_id,
        "role": message.role,
        "content": message.content,
        "citations": citations,
        "meta": meta,
        "created_at": _isoformat(message.created_at),
    }


def _session_last_activity(session_obj: models.Session, db_session) -> datetime | None:
    last_message_at = db_session.scalar(
        select(func.max(models.Message.created_at)).where(
            models.Message.session_id == session_obj.id
        )
    )
    return last_message_at or session_obj.updated_at


def _serialize_session(session_obj: models.Session, db_session) -> dict[str, Any]:
    last_activity = _session_last_activity(session_obj, db_session)
    return {
        "id": session_obj.id,
        "title": session_obj.title,
        "topic": session_obj.title,
        "updated_at": _isoformat(session_obj.updated_at),
        "last_activity": _isoformat(last_activity),
    }


def _serialize_bookmark(bookmark: models.Bookmark) -> dict[str, Any]:
    session_obj = bookmark.session
    message_obj = bookmark.message
    return {
        "id": bookmark.id,
        "session_id": bookmark.session_id,
        "message_id": bookmark.message_id,
        "created_at": _isoformat(bookmark.created_at),
        "tag": bookmark.tag,
        "session": (
            {
                "id": session_obj.id,
                "title": session_obj.title,
                "updated_at": _isoformat(session_obj.updated_at),
            }
            if session_obj
            else None
        ),
        "message": _serialize_message(message_obj) if message_obj else None,
    }


def _ensure_owner(db_session, user_profile: Mapping[str, Any]) -> models.User:
    email = user_profile.get("email")
    if not isinstance(email, str) or not email:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid user profile")
    normalized_email = email.strip().lower()
    repo = repositories.UserRepository(db_session)
    user = repo.get_by_email(normalized_email)
    if user is None:
        name = user_profile.get("name")
        user = repo.create(email=normalized_email, name=str(name) if name else None)
    else:
        name = user_profile.get("name")
        if isinstance(name, str) and name and name != (user.name or ""):
            user.name = name
    return user


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next: Callable[[Request], Any]) -> Response:
    path = request.url.path
    method = request.method
    start = time.perf_counter()
    try:
        retry_after = _enforce_limit(_GLOBAL_RATE_LIMIT, request)
        if retry_after is not None:
            duration = time.perf_counter() - start
            response = _rate_limit_response(retry_after)
            REQUEST_COUNTER.labels(path=path, method=method, status="429").inc()
            REQUEST_LATENCY.labels(path=path, method=method).observe(duration)
            return response
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start
        ERROR_COUNTER.labels(path=path, method=method).inc()
        REQUEST_COUNTER.labels(path=path, method=method, status="500").inc()
        REQUEST_LATENCY.labels(path=path, method=method).observe(duration)
        raise

    duration = time.perf_counter() - start
    status_code = getattr(response, "status_code", 500)
    REQUEST_COUNTER.labels(
        path=path,
        method=method,
        status=str(status_code),
    ).inc()
    REQUEST_LATENCY.labels(path=path, method=method).observe(duration)
    if status_code >= 500:
        ERROR_COUNTER.labels(path=path, method=method).inc()
    return response


class QueryRequest(BaseModelProto):
    q: str
    top_k: int = 3


class LoginRequest(BaseModelProto):
    email: str
    password: str


def _find_indexed_file(*args: Any, **kwargs: Any) -> Any:
    from core.index import find_indexed_file as _find

    return _find(*args, **kwargs)


def _list_indexed_files(*args: Any, **kwargs: Any) -> list[IndexedFileType]:
    from core.index import list_indexed_files as _list

    return _list(*args, **kwargs)


def _index_path(*args: Any, **kwargs: Any) -> IndexResultType:
    from core.index import index_path as _index

    return _index(*args, **kwargs)


def _reindex_existing_file(*args: Any, **kwargs: Any) -> IndexResultType:
    from core.index import reindex_existing_file as _reindex

    return _reindex(*args, **kwargs)


def _update_indexed_path(*args: Any, **kwargs: Any) -> None:
    from core.index import update_indexed_file_path as _update

    _update(*args, **kwargs)


def _llm_error_payload(exc: LLMError) -> tuple[int, dict[str, Any]]:
    if isinstance(exc, LLMTimeoutError):
        return (
            status.HTTP_504_GATEWAY_TIMEOUT,
            {"error": {"type": "timeout", "message": str(exc)}},
        )
    return (
        status.HTTP_502_BAD_GATEWAY,
        {"error": {"type": "llm_error", "message": str(exc)}},
    )


def _llm_error_response(exc: LLMError) -> JSONResponse:
    status_code, payload = _llm_error_payload(exc)
    return JSONResponse(status_code=status_code, content=payload)


def _serialize_index_result(result: IndexResultType) -> dict[str, Any]:
    return {
        "new": result.new,
        "skipped": result.skipped,
        "collection": result.collection,
        "book_id": result.book_id,
        "version": result.version,
        "file_hash": result.file_hash,
        "indexed_chunks": result.indexed_chunks,
    }


def _is_newer_version(candidate: str | None, current: str | None) -> bool:
    if not candidate:
        return False
    if not current:
        return True
    return candidate > current


def _book_title_from_meta(meta: Mapping[str, Any] | None, fallback: str) -> str:
    if isinstance(meta, Mapping):
        for key in ("title", "subject", "author"):
            value = meta.get(key)
            if isinstance(value, str):
                title = value.strip()
                if title:
                    return title
    return fallback


def _split_book_filter(book_id: str | None) -> list[str]:
    if not book_id:
        return []
    separators = [",", ";"]
    raw_value = book_id
    for sep in separators:
        raw_value = raw_value.replace(sep, ",")
    values = []
    for part in raw_value.split(","):
        candidate = part.strip()
        if candidate:
            values.append(candidate)
    return values


def _is_default_session_title(title: str | None) -> bool:
    if not title:
        return True
    normalized = title.strip().lower()
    if not normalized:
        return True
    return normalized in {"new session", "untitled session"} or normalized.startswith("session ")


_TITLE_PROMPT = (
    "You are KetabMind's naming assistant. "
    "Given a user's latest question and the assistant's reply, "
    "return a concise session title (max 6 words, title case, no punctuation).\n"
    "Question: {question}\n"
    "Answer: {answer}\n"
    "Title:"
)


def _clean_generated_title(text: str) -> str:
    cleaned = text.strip().splitlines()[0] if text else ""
    cleaned = cleaned.strip("\"'“”‘’")
    cleaned = re.sub(r"[.:]+$", "", cleaned).strip()
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip()
    return cleaned


def _suggest_session_title(question: str, answer: str) -> str | None:
    if not question and not answer:
        return None
    prompt = _TITLE_PROMPT.format(question=question.strip(), answer=answer.strip())
    try:
        llm = get_llm()
        raw = llm.generate(prompt, stream=False)
    except Exception:
        raw = ""
    text = raw if isinstance(raw, str) else str(raw)
    cleaned = _clean_generated_title(text)
    if cleaned:
        return cleaned
    fallback_source = question or answer
    tokens = fallback_source.strip().split()
    if not tokens:
        return None
    return " ".join(tokens[:6]).title()


def _maybe_autoname_session(
    session_repo: repositories.SessionRepository,
    message_repo: repositories.MessageRepository,
    session_obj: models.Session,
    message_obj: models.Message,
) -> None:
    if message_obj.role != "assistant":
        return
    if not _is_default_session_title(session_obj.title):
        return
    if message_repo.has_prior_assistant_message(message_obj):
        return
    question_obj = message_repo.get_previous_user_message(message_obj)
    suggestion = _suggest_session_title(
        question_obj.content if question_obj else "",
        message_obj.content or "",
    )
    if not suggestion:
        return
    session_repo.rename(session_obj.id, suggestion)


def _ensure_upload_root() -> Path:
    upload_dir = settings.upload_dir
    if not upload_dir:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload directory not configured",
        )
    root = Path(upload_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    try:
        return root.resolve()
    except FileNotFoundError:  # pragma: no cover - defensive fallback
        return root


async def _write_upload_to_tempfile(file: UploadFile, filename: Path) -> Path:
    suffix = filename.suffix if filename.suffix else ""
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
    close = getattr(file, "close", None)
    if close:
        result = close()
        if inspect.isawaitable(result):
            await result
    return Path(tmp.name)


def _store_uploaded_file(src: Path, owner_id: str, book_id: str, filename: str) -> Path:
    root = _ensure_upload_root()
    target_dir = root / owner_id / book_id
    target_dir.mkdir(parents=True, exist_ok=True)
    dest_path = target_dir / filename
    if dest_path.exists():
        dest_path.unlink()
    shutil.move(str(src), dest_path)
    return dest_path


def _resolve_book_file(owner_id: str, book_id: str) -> Path | None:
    root = _ensure_upload_root()
    book_dir = root / owner_id / book_id
    if not book_dir.is_dir():
        return None
    files = sorted(p for p in book_dir.iterdir() if p.is_file())
    if not files:
        return None
    return files[0]


def _encode_view_token(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    signature = hmac.new(
        settings.jwt_secret.encode("utf-8"),
        raw,
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded}.{signature}"


def _decode_view_token(token: str) -> dict[str, Any]:
    try:
        encoded, signature = token.rsplit(".", 1)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid token format") from exc
    padding = "=" * (-len(encoded) % 4)
    try:
        raw = base64.urlsafe_b64decode(encoded + padding)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid token encoding") from exc
    expected = hmac.new(
        settings.jwt_secret.encode("utf-8"),
        raw,
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Invalid token signature")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid token payload") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid token payload")
    return payload


def _build_signed_view(owner_id: str, book_id: str, filename: str, page: int) -> tuple[str, int]:
    ttl = int(settings.upload_signed_url_ttl or 0)
    if ttl <= 0:
        ttl = 300
    now = int(time.time())
    expires = now + ttl
    token = _encode_view_token(
        {
            "owner_id": owner_id,
            "book_id": book_id,
            "filename": filename,
            "page": page,
            "exp": expires,
        }
    )
    return f"/static/books/{token}", expires


@_get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@_get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@_get("/books")
def list_books(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    owner_id = str(current_user.get("id") or "")
    if not owner_id:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid user profile")
    collection = settings.qdrant_collection
    try:
        indexed_files = _list_indexed_files(collection)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("books.list_failed", error=str(exc), exc_info=True)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to load books"
        ) from exc
    latest_by_book: dict[str, IndexedFileType] = {}
    for entry in indexed_files:
        book_id = str(getattr(entry, "book_id", "") or "")
        if not book_id:
            continue
        existing = latest_by_book.get(book_id)
        if existing is None or _is_newer_version(
            getattr(entry, "version", None), getattr(existing, "version", None)
        ):
            latest_by_book[book_id] = entry
    root = _ensure_upload_root()
    owner_dir = root / owner_id
    if not owner_dir.exists():
        return {"books": []}
    directories = sorted(
        (p for p in owner_dir.iterdir() if p.is_dir()), key=lambda path: path.name.lower()
    )
    books: list[dict[str, Any]] = []
    for book_path in directories:
        book_id = book_path.name
        entry = latest_by_book.get(book_id)
        raw_meta = getattr(entry, "metadata", None) if entry is not None else None
        metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else None
        version = getattr(entry, "version", None) if entry is not None else None
        file_hash = getattr(entry, "file_hash", None) if entry is not None else None
        indexed_chunks = getattr(entry, "indexed_chunks", None) if entry is not None else None
        books.append(
            {
                "book_id": book_id,
                "title": _book_title_from_meta(metadata, book_id),
                "metadata": metadata,
                "version": version,
                "file_hash": file_hash,
                "indexed_chunks": indexed_chunks,
            }
        )
    return {"books": books}


def _csrf_headers(request: Request) -> dict[str, str]:
    token = request.cookies.get("access_token") if hasattr(request, "cookies") else None
    if isinstance(token, str) and token:
        return {"x-csrf-token": token}
    return {}


@_post("/auth/login")
def login(req: LoginRequest) -> JSONResponse:
    record = authenticate_user(req.email, req.password)
    token = create_access_token({"sub": record["id"], "email": record["email"]})
    profile = get_user_profile(record["id"])
    max_age = settings.jwt_expiration_seconds
    cookie_parts = [f"access_token={token}", "HttpOnly", "Path=/", "SameSite=Lax"]
    if max_age:
        cookie_parts.append(f"Max-Age={int(max_age)}")
    headers = {
        "set-cookie": "; ".join(cookie_parts),
        "x-csrf-token": token,
    }
    return JSONResponse({"user": profile}, headers=headers)


@_get("/me")
def me(
    request: Request,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> JSONResponse:
    headers = _csrf_headers(request)
    return JSONResponse(current_user, headers=headers)


@_post("/auth/logout")
def logout() -> JSONResponse:
    response = JSONResponse({"status": "ok"}, headers={"x-csrf-token": ""})
    response.delete_cookie("access_token", path="/")
    return response


@_get("/search")
def search_pages(
    query: Annotated[str, Query(..., min_length=1)],
    book_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int | None, Query(ge=1, le=100)] = None,
    _user: Annotated[dict[str, Any], Depends(get_current_user)] | None = None,
) -> dict[str, Any]:
    backend = get_fts_backend()
    if not backend.is_available():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Full-text search backend not configured",
        )
    page_limit = int(limit) if limit is not None else settings.fts_page_limit
    book_filters = _split_book_filter(book_id)
    try:
        matches: list[FTSMatch] | Sequence[FTSMatch]
        if book_filters:
            per_book_limit = max(1, math.ceil(page_limit / max(1, len(book_filters))))
            combined: list[FTSMatch] = []
            seen: set[tuple[str, int]] = set()
            for book_filter in book_filters:
                subset = backend.search(
                    query,
                    book_id=book_filter,
                    limit=per_book_limit,
                )
                for match in subset:
                    key = (match.book_id, match.page_num)
                    if key in seen:
                        continue
                    seen.add(key)
                    combined.append(match)
            combined.sort(key=lambda item: item.score, reverse=True)
            matches = combined[:page_limit]
        else:
            matches = backend.search(
                query,
                limit=max(1, page_limit),
                book_id=None,
            )
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("search.fts_failed", error=str(exc), exc_info=True)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search backend failure",
        ) from exc

    results = [
        {
            "book_id": match.book_id,
            "page_num": match.page_num,
            "text": match.text,
            "score": match.score,
        }
        for match in matches
    ]
    return {"results": results}


@_post("/query")
def query(
    req: QueryRequest,
    request: Request,
    _user: Annotated[dict[str, Any], Depends(get_current_user)],
    stream: Annotated[bool, Query()] = False,
    debug: Annotated[bool, Query()] = False,
) -> Any:
    retry_after = _enforce_limit(_QUERY_RATE_LIMIT, request)
    if retry_after is not None:
        return _rate_limit_response(retry_after)
    log.info("query.received", stream=stream, top_k=req.top_k, debug=debug)
    if stream:

        def gen() -> Iterable[str]:
            log.info("query.stream.start", top_k=req.top_k, debug=debug)
            try:
                chunks = stream_answer(req.q, top_k=req.top_k)
            except LLMError as exc:
                log.warning("query.stream.llm_error", error=str(exc), exc_info=str(exc))
                yield json.dumps({"error": str(exc)}) + "\n"
                return
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("query.stream.failure", error=str(exc))
                yield json.dumps({"error": str(exc)}) + "\n"
                return

            try:
                for chunk in chunks:
                    yield json.dumps(chunk) + "\n"
            except LLMError as exc:
                log.warning("query.stream.llm_error", error=str(exc), exc_info=str(exc))
                yield json.dumps({"error": str(exc)}) + "\n"
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("query.stream.failure", error=str(exc))
                yield json.dumps({"error": str(exc)}) + "\n"
            else:
                log.info("query.stream.complete")

        return StreamingResponse(gen(), media_type="application/json")
    try:
        result = answer(req.q, top_k=req.top_k)
        log.info("query.complete", stream=False, top_k=req.top_k, debug=debug)
        payload = build_query_response(result, debug=debug)
        meta = payload.get("meta") if isinstance(payload, dict) else None
        if isinstance(meta, dict):
            token_usage = meta.get("token_usage")
            if token_usage is not None:
                with suppress(TypeError, ValueError):  # pragma: no cover - defensive
                    TOKEN_USAGE_GAUGE.set(float(token_usage))
        return payload
    except LLMError as exc:
        log.warning("query.llm_error", error=str(exc), exc_info=str(exc))
        return _llm_error_response(exc)


@_post("/index")
def index(
    req: IndexRequest,
    _user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    collection = req.collection or settings.qdrant_collection
    owner_id = str(_user.get("id") or "")
    if not owner_id:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid user profile")

    if req.path:
        path = Path(req.path)
        if not path.exists():
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"File not found: {path}")
        file_hash = sha256_file(path)
        existing = _find_indexed_file(collection, file_hash)
        if existing:
            return {
                "new": 0,
                "skipped": existing.indexed_chunks,
                "collection": collection,
                "book_id": existing.book_id,
                "version": existing.version,
                "file_hash": existing.file_hash,
                "indexed_chunks": existing.indexed_chunks,
            }
        try:
            result = _index_path(
                path,
                collection=collection,
                file_hash=file_hash,
                metadata=req.metadata(),
            )
        except SystemExit as exc:  # invalid path or type
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return _serialize_index_result(result)

    if not req.file_hash:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="file_hash is required when path is not provided",
        )

    existing = _find_indexed_file(collection, req.file_hash)
    if not existing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Indexed file not found")

    if req.book_id and str(req.book_id) != existing.book_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Book ID does not match stored record",
        )

    stored_path = existing.path
    if stored_path is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="Stored file path unavailable for reindexing",
        )

    root = _ensure_upload_root()
    stored_path = stored_path.resolve()
    try:
        stored_path.relative_to(root / owner_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Forbidden") from exc

    if not stored_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Stored file not found")

    request_meta = req.metadata()
    merged_meta: dict[str, Any] = {}
    if isinstance(existing.metadata, dict):
        merged_meta.update(existing.metadata)
    if request_meta:
        merged_meta.update(request_meta)

    meta_arg = merged_meta if merged_meta else None

    try:
        result = _reindex_existing_file(
            collection=collection,
            file_hash=req.file_hash,
            metadata=meta_arg,
        )
    except SystemExit as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return _serialize_index_result(result)


def _upload_metadata(
    author: Annotated[str | None, Form()] = None,
    year: Annotated[str | None, Form()] = None,
    subject: Annotated[str | None, Form()] = None,
    title: Annotated[str | None, Form()] = None,
) -> Metadata:
    return Metadata(author=author, year=year, subject=subject, title=title)


@_post("/upload")
async def upload(
    file: Annotated[UploadFile, File(...)],
    meta: Annotated[Metadata, Depends(_upload_metadata)],
    _user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    filename = Path(file.filename or "upload")
    tmp_path = await _write_upload_to_tempfile(file, filename)
    owner_id = str(_user.get("id") or "")
    if not owner_id:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid user profile")
    meta_payload = meta.as_dict()
    response_payload: dict[str, Any]
    try:
        collection = settings.qdrant_collection
        file_hash = sha256_file(tmp_path)
        existing = _find_indexed_file(collection, file_hash)
        if existing:
            book_id = existing.book_id
            version = existing.version
            stored_path = _store_uploaded_file(tmp_path, owner_id, book_id, filename.name)
            _update_indexed_path(collection, file_hash, stored_path)
            response_payload = {
                "book_id": book_id,
                "version": version,
                "file_hash": file_hash,
                "path": str(stored_path),
                "already_indexed": True,
            }
        else:
            result = _index_path(
                tmp_path,
                collection=collection,
                file_hash=file_hash,
                metadata=meta_payload,
            )
            book_id = result.book_id
            version = result.version
            file_hash = result.file_hash
            stored_path = _store_uploaded_file(tmp_path, owner_id, book_id, filename.name)
            _update_indexed_path(collection, file_hash, stored_path)
            response_payload = {
                "book_id": book_id,
                "version": version,
                "file_hash": file_hash,
                "path": str(stored_path),
                "indexed_chunks": result.indexed_chunks,
            }
        if meta_payload:
            response_payload["meta"] = meta_payload
    except SystemExit as exc:  # invalid path or type
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        with suppress(FileNotFoundError):
            tmp_path.unlink()

    return response_payload


@_get("/book/{book_id}/page/{page}/view")
def book_page_view(
    book_id: str,
    page: int,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    try:
        page_num = int(page)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid page number") from exc
    if page_num < 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid page number")
    owner_id = str(current_user.get("id") or "")
    if not owner_id:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid user profile")
    file_path = _resolve_book_file(owner_id, book_id)
    if file_path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Book not found")
    url, expires_at = _build_signed_view(owner_id, book_id, file_path.name, page_num)
    return {"url": url, "expires_at": expires_at}


@_get("/static/books/{token}")
def serve_book_asset(token: str) -> Response:
    payload = _decode_view_token(token)
    try:
        expires_at = int(payload.get("exp", 0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid token payload") from exc
    if expires_at < int(time.time()):
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="URL expired")
    owner_id = str(payload.get("owner_id") or "")
    book_id = str(payload.get("book_id") or "")
    filename = str(payload.get("filename") or "")
    page = payload.get("page")
    if not owner_id or not book_id or not filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid token payload")
    if not isinstance(page, int) or page < 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Invalid token payload")
    file_path = _ensure_upload_root() / owner_id / book_id / filename
    if not file_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="File not found")
    media_type, _ = mimetypes.guess_type(str(file_path))
    headers = {
        "Cache-Control": "private, max-age=0, no-store",
        "content-type": media_type or "application/octet-stream",
    }
    if filename:
        headers["content-disposition"] = f"inline; filename={filename}"
    return Response(file_path.read_bytes(), headers=headers)


@_get("/bookmarks")
def bookmarks(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    user_id: Annotated[str | None, Query()] = None,
    tag: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    target_id = user_id or current_user["id"]
    if target_id != current_user["id"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Forbidden")
    with session_scope() as db_session:
        owner = _ensure_owner(db_session, current_user)
        repo = repositories.BookmarkRepository(db_session, owner.id)
        filter_tag = (tag or "").strip() or None
        records = repo.list(tag=filter_tag)
        payload = [_serialize_bookmark(record) for record in records]
        return {"bookmarks": payload}


@_post("/bookmarks")
def create_bookmark(
    payload: BookmarkCreate,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    with session_scope() as db_session:
        owner = _ensure_owner(db_session, current_user)
        repo = repositories.BookmarkRepository(db_session, owner.id)
        message_id = _parse_identifier(payload.message_id, field="message_id")
        tag_value = (payload.tag or "").strip() or None
        try:
            bookmark = repo.create(message_id=message_id, tag=tag_value)
        except ValueError as exc:
            detail = str(exc)
            status_code = status.HTTP_404_NOT_FOUND
            if "assistant" in detail.lower():
                status_code = status.HTTP_400_BAD_REQUEST
            raise HTTPException(status_code, detail) from exc
        db_session.refresh(bookmark)
        return {"bookmark": _serialize_bookmark(bookmark)}


@_delete("/bookmarks/{bookmark_id}")
def delete_bookmark(
    bookmark_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> Response:
    bookmark_pk = _parse_identifier(bookmark_id, field="bookmark_id")
    with session_scope() as db_session:
        owner = _ensure_owner(db_session, current_user)
        repo = repositories.BookmarkRepository(db_session, owner.id)
        if not repo.delete(bookmark_pk):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Bookmark not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@_get("/sessions")
def sessions(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    user_id: Annotated[str | None, Query()] = None,
    query: Annotated[str | None, Query()] = None,
    sort: Annotated[str, Query()] = "date_desc",
) -> dict[str, Any]:
    target_id = user_id or current_user["id"]
    if target_id != current_user["id"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Forbidden")
    with session_scope() as db_session:
        owner = _ensure_owner(db_session, current_user)
        repo = repositories.SessionRepository(db_session, owner.id)
        allowed_sorts = {"date_desc", "date_asc", "title_asc", "title_desc"}
        sort_key = (sort or "date_desc").strip().lower()
        if sort_key not in allowed_sorts:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid sort parameter")
        search = query.strip() if isinstance(query, str) else None
        records = repo.list(
            query=search if search else None,
            sort=sort_key,
        )
        payload = [_serialize_session(record, db_session) for record in records]
        return {"sessions": payload}


@_delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> Response:
    session_pk = _parse_identifier(session_id, field="session_id")
    with session_scope() as db_session:
        owner = _ensure_owner(db_session, current_user)
        repo = repositories.SessionRepository(db_session, owner.id)
        if not repo.soft_delete(session_pk):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@_post("/sessions")
def create_session(
    payload: SessionCreate,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    with session_scope() as db_session:
        owner = _ensure_owner(db_session, current_user)
        repo = repositories.SessionRepository(db_session, owner.id)
        title = (payload.title or "New session").strip() or "New session"
        book_id: int | None = None
        if payload.book_id is not None:
            book_id = _parse_identifier(payload.book_id, field="book_id")
        try:
            session_obj = repo.create(title=title, book_id=book_id)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
        db_session.refresh(session_obj)
        return {"session": _serialize_session(session_obj, db_session)}


@_patch("/sessions/{session_id}")
def update_session(
    session_id: str,
    payload: SessionUpdate,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    session_pk = _parse_identifier(session_id, field="session_id")
    new_title = (payload.title or "").strip()
    if not new_title:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Title is required")
    with session_scope() as db_session:
        owner = _ensure_owner(db_session, current_user)
        session_repo = repositories.SessionRepository(db_session, owner.id)
        record = session_repo.rename(session_pk, new_title)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
        db_session.refresh(record)
        return {"session": _serialize_session(record, db_session)}


@_get("/sessions/{session_id}/messages")
def session_messages(
    session_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    session_pk = _parse_identifier(session_id, field="session_id")
    with session_scope() as db_session:
        owner = _ensure_owner(db_session, current_user)
        session_repo = repositories.SessionRepository(db_session, owner.id)
        session_obj = session_repo.get(session_pk)
        if session_obj is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
        message_repo = repositories.MessageRepository(db_session, owner.id)
        records = message_repo.list_for_session(session_obj.id)
        payload = [_serialize_message(message) for message in records]
        return {"messages": payload}


@_post("/sessions/{session_id}/messages")
def create_session_message(
    session_id: str,
    payload: MessageCreate,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> dict[str, Any]:
    session_pk = _parse_identifier(session_id, field="session_id")
    role = (payload.role or "").strip().lower()
    if role not in {"assistant", "user", "system"}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid message role")
    with session_scope() as db_session:
        owner = _ensure_owner(db_session, current_user)
        session_repo = repositories.SessionRepository(db_session, owner.id)
        session_obj = session_repo.get(session_pk)
        if session_obj is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
        message_repo = repositories.MessageRepository(db_session, owner.id)
        message = message_repo.create(
            session_id=session_obj.id,
            role=role,
            content=payload.content,
            citations=payload.citations,
            meta=payload.meta,
        )
        db_session.refresh(message)
        if role == "assistant":
            with suppress(Exception):
                _maybe_autoname_session(session_repo, message_repo, session_obj, message)
        return {"message": _serialize_message(message)}


@_post("/sessions/{session_id}/messages/stream")
async def stream_session_message(
    session_id: str,
    payload: StreamMessageRequest,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    book_id: Annotated[str | None, Query()] = None,
) -> StreamingResponse:
    """Stream an assistant response for the given session using server-sent events."""

    session_pk = _parse_identifier(session_id, field="session_id")
    book_ids = _split_book_filter(book_id)
    book_filter = book_ids[0] if book_ids else None
    log.info(
        "chat.stream.request",
        session_id=session_id,
        book_ids=book_ids,
        model=payload.model,
    )

    question = (payload.content or "").strip()
    session_db_id: int | None = None
    owner_id: int | None = None

    with session_scope() as db_session:
        owner = _ensure_owner(db_session, current_user)
        session_repo = repositories.SessionRepository(db_session, owner.id)
        session_obj = session_repo.get(session_pk)
        if session_obj is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
        message_repo = repositories.MessageRepository(db_session, owner.id)
        history_records = message_repo.list_for_session(session_obj.id)
        latest_user = next(
            (record for record in reversed(history_records) if record.role == "user"),
            None,
        )
        if latest_user and latest_user.content:
            question = latest_user.content.strip()
        session_db_id = session_obj.id
        owner_id = owner.id

    if not question:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "User message content is required")

    if session_db_id is None or owner_id is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Unable to load session")

    log.info(
        "chat.stream.start",
        session_id=session_id,
        book_id=book_filter,
    )

    async def _event_stream() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        final_answer: str | None = None
        final_contexts: list[dict[str, Any]] = []
        final_meta: dict[str, Any] | None = None
        error_message: str | None = None
        cancelled = False

        def _produce() -> None:
            try:
                for chunk in stream_answer(question, book_id=book_filter):
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
                    final_contexts = [
                        dict(ctx)
                        for ctx in raw_contexts
                        if isinstance(ctx, Mapping)
                    ]
                    raw_meta = chunk.get("meta")
                    final_meta = _derive_message_meta(
                        final_answer,
                        final_contexts,
                        raw_meta if isinstance(raw_meta, Mapping) else None,
                    )
                    data: dict[str, Any] = {
                        "answer": final_answer,
                        "contexts": final_contexts,
                    }
                    if final_meta:
                        data["meta"] = final_meta
                    debug_payload = chunk.get("debug")
                    if debug_payload is not None:
                        data["debug"] = debug_payload
                    yield "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
                elif kind == "error":
                    error_message = str(item)
                    log.warning(
                        "chat.stream.error",
                        session_id=session_id,
                        error=error_message,
                    )
                    yield "data: " + json.dumps({"error": error_message}, ensure_ascii=False) + "\n\n"
                    break
                elif kind == "done":
                    break
        except asyncio.CancelledError:  # pragma: no cover - client cancelled
            cancelled = True
            log.info("chat.stream.cancelled", session_id=session_id)
        finally:
            await producer_task

        if not cancelled and error_message is None and final_answer is not None:
            citations = _build_message_citations(final_contexts)
            _persist_assistant_message(
                session_db_id,
                owner_id,
                final_answer,
                citations=citations,
                meta=final_meta,
            )
            log.info(
                "chat.stream.persisted",
                session_id=session_id,
                citation_count=len(citations),
            )

        if not cancelled:
            yield "data: [DONE]\n\n"
            log.info(
                "chat.stream.complete",
                session_id=session_id,
                error=error_message,
            )

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@_post("/export")
def export_message(
    payload: ExportRequest,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
) -> Response:
    message_pk = _parse_identifier(payload.message_id, field="message_id")
    export_format = (payload.format or "pdf").strip().lower()
    allowed_formats = {"pdf", "word", "docx"}
    if export_format not in allowed_formats:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unsupported export format")

    with session_scope() as db_session:
        owner = _ensure_owner(db_session, current_user)
        message_repo = repositories.MessageRepository(db_session, owner.id)
        message_obj = message_repo.get(message_pk)
        if message_obj is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found")
        if message_obj.role != "assistant":
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Only assistant messages can be exported"
            )
        question_obj = message_repo.get_previous_user_message(message_obj)
        citations = _safe_json_list(message_obj.citations) or []
        meta = _safe_json_mapping(message_obj.meta) or {}
        answer_payload = {
            "question": question_obj.content if question_obj else "",
            "answer": message_obj.content,
            "citations": citations,
            "meta": meta,
        }
        try:
            payload_bytes = export_answer(answer_payload, format=export_format)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    if export_format == "pdf":
        media_type = "application/pdf"
        filename = "answer.pdf"
    else:
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        filename = "answer.docx"

    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return Response(payload_bytes, media_type=media_type, headers=headers)


@_get("/metrics")
def metrics() -> Response:
    payload = generate_latest().decode("utf-8")
    return Response(
        payload,
        headers={
            "content-type": CONTENT_TYPE_LATEST,
            "content-encoding": "identity",
        },
    )
