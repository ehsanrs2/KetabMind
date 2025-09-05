from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from core.answer.answerer import answer

app = FastAPI(title="KetabMind API")


class QueryRequest(BaseModel):
    q: str
    top_k: int = 3


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    return answer(req.q, top_k=req.top_k)
