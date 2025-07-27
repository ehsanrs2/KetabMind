"""FastAPI application."""
# mypy: ignore-errors

from pathlib import Path

from collections.abc import AsyncGenerator
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse

from core.ingest import pdf
from core.chunk.sliding import chunk_text
from core.embed.mock import MockEmbedder
from core.vector.qdrant import VectorStore, ChunkPayload
from core.retrieve.simple import Retriever
from core.answer.mock import generate_answer

app = FastAPI()
embedder = MockEmbedder()
store = VectorStore()
retriever = Retriever(embedder, store)


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
            "content_hash": "0",
        }
        for i, chunk in enumerate(chunks)
    ]
    store.upsert(embeddings, payloads)
    return {"status": "indexed"}


@app.post("/query")
async def query(q: str, top_k: int = 5) -> StreamingResponse:
    docs = retriever.retrieve(q, top_k)
    texts = [d["text"] for d in docs]
    answer = generate_answer(q, texts)

    async def stream() -> AsyncGenerator[str, None]:
        yield answer

    return StreamingResponse(stream(), media_type="text/plain")
