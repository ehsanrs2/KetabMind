from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from core.answer.answerer import answer, stream_answer
from core.answer.llm import LLMError, LLMTimeoutError
from core.index import index_path

if TYPE_CHECKING:

    class BaseModelProto:
        pass

else:  # pragma: no cover - runtime import
    BaseModelProto = BaseModel

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


@_get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@_post("/query")
def query(req: QueryRequest, stream: bool = Query(False)) -> Any:
    if stream:
        try:
            chunks = stream_answer(req.q, top_k=req.top_k)
        except LLMError as exc:
            return _llm_error_response(exc)

        def gen() -> Iterable[str]:
            try:
                for chunk in chunks:
                    yield json.dumps(chunk) + "\n"
            except LLMError as exc:  # pragma: no cover - defensive
                _, payload = _llm_error_payload(exc)
                payload["final"] = True
                yield json.dumps(payload) + "\n"

        return StreamingResponse(gen(), media_type="application/json")
    try:
        return answer(req.q, top_k=req.top_k)
    except LLMError as exc:
        return _llm_error_response(exc)


@_post("/index")
def index(req: IndexRequest) -> dict[str, Any]:
    try:
        new, skipped, collection = index_path(Path(req.path), collection=req.collection)
    except SystemExit as exc:  # invalid path or type
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"new": new, "skipped": skipped, "collection": collection}


@_post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    tmp_dir = Path(tempfile.gettempdir())
    filename = Path(file.filename or "upload")
    dest = tmp_dir / filename.name
    dest.write_bytes(await file.read())
    try:
        new, skipped, collection = index_path(dest)
    except SystemExit as exc:  # invalid path or type
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"new": new, "skipped": skipped, "collection": collection, "path": str(dest)}
