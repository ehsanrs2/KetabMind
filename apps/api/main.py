from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.answer.answerer import answer, stream_answer
from core.index import index_path

app = FastAPI(title="KetabMind API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    q: str
    top_k: int = 3


class IndexRequest(BaseModel):
    path: str
    collection: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest, stream: bool = False) -> Any:
    if stream:

        def gen() -> Iterable[str]:
            for chunk in stream_answer(req.q, top_k=req.top_k):
                yield json.dumps(chunk) + "\n"

        return StreamingResponse(gen(), media_type="application/json")
    return answer(req.q, top_k=req.top_k)


@app.post("/index")
def index(req: IndexRequest) -> dict[str, Any]:
    try:
        new, skipped, collection = index_path(Path(req.path), collection=req.collection)
    except SystemExit as exc:  # invalid path or type
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"new": new, "skipped": skipped, "collection": collection}


@app.post("/upload")
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
