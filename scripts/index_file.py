"""Index a file into Qdrant."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Optional, cast

import typer

from core.chunk.sliding import chunk_text
from core.embed.mock import MockEmbedder
from core.vector.qdrant import VectorStore, ChunkPayload
from core.ingest import pdf, epub

app = typer.Typer()


def _read_lines(path: Path) -> Iterable[str]:
    if path.suffix.lower() == ".pdf":
        return (p["text"] for p in pdf.extract_pages(path))
    if path.suffix.lower() == ".epub":
        return (p["text"] for p in epub.extract_pages(path))
    return path.read_text(encoding="utf-8").splitlines()


@app.command()
def main(in_path: Path, collection: str = "books") -> None:
    """Index a single file."""
    embedder = MockEmbedder()
    store = VectorStore()
    store.collection = collection
    lines = list(_read_lines(in_path))
    chunks = chunk_text(lines)
    embeddings = embedder.embed(chunks)
    payloads: List[ChunkPayload] = []
    for i, chunk in enumerate(chunks):
        payloads.append(
            cast(
                ChunkPayload,
                {
                    "text": chunk,
                    "book_id": in_path.name,
                    "chapter": None,
                    "page_start": 0,
                    "page_end": 0,
                    "chunk_id": str(i),
                    "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                },
            )
        )
    new, skipped = store.upsert(embeddings, payloads)
    typer.echo(f"new={new} skipped={skipped}")


if __name__ == "__main__":
    app()

