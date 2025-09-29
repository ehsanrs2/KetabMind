from __future__ import annotations

import json
import uuid
import tempfile
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import structlog
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from structlog.contextvars import bind_contextvars, clear_contextvars

from core.answer.answerer import answer, stream_answer
from core.answer.llm import LLMError, LLMTimeoutError
from core.config import settings
from core.index import IndexResult, find_indexed_file, index_path
from core.logging import configure_logging
from utils.hash import sha256_file

if TYPE_CHECKING:

    class BaseModelProto:
        pass

else:  # pragma: no cover - runtime import
    BaseModelProto = BaseModel

configure_logging()

log = structlog.get_logger(__name__)

app = FastAPI(title="KetabMind API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

F = TypeVar("F", bound=Callable[..., Any])


def _get(path: str) -> Callable[[F], F]:
    return cast(Callable[[F], F], app.get(path))


def _post(path: str) -> Callable[[F], F]:
    return cast(Callable[[F], F], app.post(path))


class QueryRequest(BaseModelProto):
    q: str
    top_k: int = 3


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


class IndexRequest(BaseModelProto):
    path: str
    collection: str | None = None


def _serialize_index_result(result: IndexResult) -> dict[str, Any]:
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


@_post("/query")
def query(req: QueryRequest, request: Request, stream: bool = Query(False)) -> Any:
    request_id = getattr(request.state, "request_id", None)
    log.info(
        "query.received",
        stream=stream,
        top_k=req.top_k,
        request_id=request_id,
    )
    if stream:
        def gen() -> Iterable[str]:
            if request_id is not None:
                clear_contextvars()
                bind_contextvars(request_id=request_id)
            log.info("query.stream.start", top_k=req.top_k, request_id=request_id)
            try:
                chunks = stream_answer(req.q, top_k=req.top_k)
            except LLMError as exc:
                log.warning(
                    "query.stream.llm_error",
                    error=str(exc),
                    request_id=request_id,
                )
                yield json.dumps({"error": str(exc)}) + "\n"
                return
            except Exception as exc:  # pragma: no cover - defensive
                log.exception(
                    "query.stream.failure",
                    error=str(exc),
                    request_id=request_id,
                )
                yield json.dumps({"error": str(exc)}) + "\n"
                return

            try:
                for chunk in chunks:
                    yield json.dumps(chunk) + "\n"
            except LLMError as exc:
                log.warning(
                    "query.stream.llm_error",
                    error=str(exc),
                    request_id=request_id,
                )
                yield json.dumps({"error": str(exc)}) + "\n"
            except Exception as exc:  # pragma: no cover - defensive
                log.exception(
                    "query.stream.failure",
                    error=str(exc),
                    request_id=request_id,
                )
                yield json.dumps({"error": str(exc)}) + "\n"
            else:
                log.info("query.stream.complete", request_id=request_id)
            finally:
                clear_contextvars()

        return StreamingResponse(gen(), media_type="application/json")
    try:
        result = answer(req.q, top_k=req.top_k)
        log.info(
            "query.complete",
            stream=False,
            top_k=req.top_k,
            request_id=request_id,
        )
        return result
    except LLMError as exc:
        log.warning("query.llm_error", error=str(exc), request_id=request_id)
        return _llm_error_response(exc)


@_post("/index")
def index(req: IndexRequest) -> dict[str, Any]:
    try:
        path = Path(req.path)
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        collection = req.collection or settings.qdrant_collection
        file_hash = sha256_file(path)
        existing = find_indexed_file(collection, file_hash)
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
        result = index_path(path, collection=collection, file_hash=file_hash)
    except SystemExit as exc:  # invalid path or type
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_index_result(result)


@_post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    tmp_dir = Path(tempfile.gettempdir())
    filename = Path(file.filename or "upload")
    dest = tmp_dir / filename.name
    dest.write_bytes(await file.read())
    try:
        collection = settings.qdrant_collection
        file_hash = sha256_file(dest)
        existing = find_indexed_file(collection, file_hash)
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
        result = index_path(dest, collection=collection, file_hash=file_hash)
    except SystemExit as exc:  # invalid path or type
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = _serialize_index_result(result)
    payload["path"] = str(dest)
    return payload


@app.middleware("http")
async def add_request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Any]]
) -> Any:
    clear_contextvars()
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    bind_contextvars(request_id=request_id)
    request.state.request_id = request_id
    log.info(
        "request.start",
        method=request.method,
        path=request.url.path,
        request_id=request_id,
    )
    try:
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info(
            "request.finish",
            status_code=response.status_code,
            request_id=request_id,
        )
        return response
    except Exception as exc:  # pragma: no cover - defensive
        log.exception(
            "request.error",
            error=str(exc),
            method=request.method,
            path=request.url.path,
            request_id=request_id,
        )
        raise
    finally:
        clear_contextvars()
