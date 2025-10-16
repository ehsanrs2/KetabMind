import hashlib
from pathlib import Path
from typing import cast

import pytest

import core.vector.qdrant as qdrant
from core.chunk.sliding import chunk_text
from core.embed.mock import MockEmbedder
from core.ingest.pdf import extract_pages
from core.vector.qdrant import ChunkPayload, VectorStore
from qdrant_client.http import models as rest


def index_pdf(path: Path, store: VectorStore) -> tuple[int, int]:
    lines = [p["text"] for p in extract_pages(path)]
    chunks = chunk_text(lines)
    embeddings = MockEmbedder().embed(chunks)
    payloads = [
        cast(
            ChunkPayload,
            {
                "text": c,
                "book_id": path.name,
                "chapter": None,
                "page_start": 0,
                "page_end": 0,
                "chunk_id": str(i),
                "content_hash": hashlib.sha256(c.encode("utf-8")).hexdigest(),
            },
        )
        for i, c in enumerate(chunks)
    ]
    return store.upsert(embeddings, payloads)


def test_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qdrant.settings, "qdrant_mode", "local")  # type: ignore[attr-defined]
    monkeypatch.setattr(
        qdrant.settings,
        "qdrant_location",
        ":memory:",
    )  # type: ignore[attr-defined]
    store = VectorStore()
    store.client.recreate_collection(
        collection_name=store.collection,
        vectors_config=rest.VectorParams(size=3, distance=rest.Distance.COSINE),
    )
    path = Path("docs/fixtures/sample.pdf")
    index_pdf(path, store)
    new, skipped = index_pdf(path, store)
    assert skipped > 0
