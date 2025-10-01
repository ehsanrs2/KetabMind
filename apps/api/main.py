from __future__ import annotations

import gzip
import json
import math
import tempfile
import time
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import structlog
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware import RequestIDMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from limits import RateLimitItemPerSecond
from slowapi import Limiter
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from utils.pydantic_compat import BaseModel

from apps.api.routes.query import build_query_response
from apps.api.schemas import IndexRequest, Metadata
from core.answer.answerer import answer, stream_answer
from core.answer.llm import LLMError, LLMTimeoutError
from core.config import settings
from utils.logging import configure_logging
from utils.hash import sha256_file

if TYPE_CHECKING:

    from core.index import IndexResult as IndexResultType

    class BaseModelProto:
        pass

else:  # pragma: no cover - runtime import
    IndexResultType = Any
    BaseModelProto = BaseModel

configure_logging()

log = structlog.get_logger(__name__)


def _client_identifier(request: Request) -> str:
    headers = getattr(request, "headers", {})
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    real_ip = headers.get("x-real-ip")
    if real_ip:
        return real_ip
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client else None
    if host:
        return host
    return "127.0.0.1"


class _GZipMiddleware:
    def __init__(self, app: Any, minimum_size: int = 500) -> None:
        self.app = app
        self.minimum_size = minimum_size

    async def __call__(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        response = await call_next(request)
        body = getattr(response, "_content", None)
        if body is None:
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
        response.headers.setdefault("content-encoding", "gzip")
        response.headers.setdefault("vary", "Accept-Encoding")
        response.headers["content-length"] = str(len(compressed))
        response._content = compressed  # type: ignore[attr-defined]
        return response


app = FastAPI(title="KetabMind API")

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
if not hasattr(app, "state"):
    app.state = SimpleNamespace()
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

if "REQUEST_COUNTER" not in globals():
    REQUEST_COUNTER = Counter(
        "api_requests_total",
        "Total number of API requests",
        labelnames=("path", "method", "status"),
    )
if "REQUEST_LATENCY" not in globals():
    REQUEST_LATENCY = Histogram(
        "api_request_latency_seconds",
        "Latency of API requests in seconds",
        labelnames=("path", "method"),
    )
if "ERROR_COUNTER" not in globals():
    ERROR_COUNTER = Counter(
        "api_errors_total",
        "Total number of API errors",
        labelnames=("path", "method"),
    )
if "TOKEN_USAGE_GAUGE" not in globals():
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


def _enforce_limit(
    limit: RateLimitItemPerSecond | None, request: Request
) -> int | None:
    if not limiter.enabled or limit is None:
        return None
    key = limiter._key_func(request) if hasattr(limiter, "_key_func") else _client_identifier(request)
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


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next: Callable[[Request], Any]):
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



def _find_indexed_file(*args: Any, **kwargs: Any) -> Any:
    from core.index import find_indexed_file as _find

    return _find(*args, **kwargs)


def _index_path(*args: Any, **kwargs: Any) -> IndexResultType:
    from core.index import index_path as _index

    return _index(*args, **kwargs)


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


@_get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@_get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@_post("/query")
def query(
    req: QueryRequest,
    request: Request,
    stream: bool = Query(False),
    debug: bool = Query(False),
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
                try:
                    TOKEN_USAGE_GAUGE.set(float(token_usage))
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    pass
        return payload
    except LLMError as exc:
        log.warning("query.llm_error", error=str(exc), exc_info=str(exc))
        return _llm_error_response(exc)


@_post("/index")
def index(req: IndexRequest) -> dict[str, Any]:
    try:
        path = Path(req.path)
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        collection = req.collection or settings.qdrant_collection
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
        result = _index_path(
            path,
            collection=collection,
            file_hash=file_hash,
            metadata=req.metadata(),
        )
    except SystemExit as exc:  # invalid path or type
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_index_result(result)


def _upload_metadata(
    author: str | None = Form(None),
    year: str | None = Form(None),
    subject: str | None = Form(None),
) -> Metadata:
    return Metadata(author=author, year=year, subject=subject)


@_post("/upload")
async def upload(
    file: UploadFile = File(...),  # noqa: B008
    meta: Metadata = Depends(_upload_metadata),
) -> dict[str, Any]:
    tmp_dir = Path(tempfile.gettempdir())
    filename = Path(file.filename or "upload")
    dest = tmp_dir / filename.name
    dest.write_bytes(await file.read())
    try:
        collection = settings.qdrant_collection
        file_hash = sha256_file(dest)
        existing = _find_indexed_file(collection, file_hash)
        if existing:
            payload = {
                "new": 0,
                "skipped": existing.indexed_chunks,
                "collection": collection,
                "book_id": existing.book_id,
                "version": existing.version,
                "file_hash": existing.file_hash,
                "indexed_chunks": existing.indexed_chunks,
                "path": str(dest),
            }
            return payload
        result = _index_path(
            dest,
            collection=collection,
            file_hash=file_hash,
            metadata=meta.as_dict(),
        )
    except SystemExit as exc:  # invalid path or type
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = _serialize_index_result(result)
    payload["path"] = str(dest)
    return payload


@_get("/metrics")
def metrics() -> Response:
    return Response(generate_latest().decode("utf-8"), headers={"content-type": CONTENT_TYPE_LATEST})


