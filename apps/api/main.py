from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.answer.answerer import answer

app = FastAPI(title="KetabMind API")


class QueryRequest(BaseModel):
    q: str
    top_k: int = 3


class Context(BaseModel):
    text: str
    book_id: str
    page_start: int
    page_end: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    contexts: list[Context]
    debug: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    res = answer(req.q, top_k=req.top_k)
    contexts = [Context(**c) for c in res.get("contexts", [])]
    return QueryResponse(
        answer=res.get("answer", ""),
        contexts=contexts,
        debug={"count": len(contexts)},
    )
