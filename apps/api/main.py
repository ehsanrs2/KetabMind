"""FastAPI application."""
# mypy: ignore-errors

from pathlib import Path
import hashlib

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from core.ingest import pdf
from core.chunk.sliding import chunk_text
from core.embed import get_embedder
from core.vector.qdrant import VectorStore, ChunkPayload
from core.answer.answerer import answer

app = FastAPI()
embedder = get_embedder()
store = VectorStore()


@app.post("/index")
async def index(file: UploadFile) -> dict[str, str]:
    filename = file.filename or "upload"
    path = Path("/tmp") / filename
    with path.open("wb") as f:
        f.write(await file.read())
    lines = pdf.extract_text(path)
    chunks = chunk_text(lines)
    embeddings = embedder.embed(chunks)
    payloads: list[ChunkPayload] = [
        {
            "text": chunk,
            "book_id": filename,
            "chapter": None,
            "page_start": 0,
            "page_end": 0,
            "chunk_id": str(i),
            "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
        }
        for i, chunk in enumerate(chunks)
    ]
    store.upsert(embeddings, payloads)
    return {"status": "indexed"}


class QueryRequest(BaseModel):
    q: str
    top_k: int = 8


@app.post("/query")
async def query(request: QueryRequest) -> dict:
    return answer(request.q, request.top_k)
