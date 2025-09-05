from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.answer.answerer import answer, stream_answer
from core.index import index_path

app = FastAPI(title="KetabMind API")


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
