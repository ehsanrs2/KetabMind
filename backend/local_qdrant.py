"""Offline-friendly helpers for working with a local Qdrant instance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from core.chunk.chunker import Chunk, sliding_window_chunks
from core.config import get_settings
from core.embed import get_embedder
from core.ingest.pdf import extract_pages

logger = structlog.get_logger(__name__)

_CLIENT: QdrantClient | None = None


def _storage_path() -> Path:
    """Return the filesystem path used for local Qdrant storage."""

    path = Path("~/.ketabmind/qdrant").expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_client() -> QdrantClient:
    """Return a singleton Qdrant client stored on disk."""

    global _CLIENT
    if _CLIENT is None:
        storage = _storage_path()
        logger.debug("local_qdrant.init", storage=str(storage))
        _CLIENT = QdrantClient(path=str(storage))
    return _CLIENT


def _current_vector_size(collection: str) -> int | None:
    client = get_client()
    try:
        info = client.get_collection(collection_name=collection)
    except Exception:
        return None
    params = getattr(getattr(info, "config", None), "params", None)
    if isinstance(params, rest.CollectionParams):
        vectors = params.vectors
        if isinstance(vectors, rest.VectorParams):
            return int(vectors.size)
        if isinstance(vectors, dict):
            size = vectors.get("size")
            if size is not None:
                return int(size)
    return None


def ensure_collection(collection: str, vector_size: int) -> None:
    """Ensure the Qdrant collection exists with the requested dimensionality."""

    client = get_client()
    current = _current_vector_size(collection)
    if current is None:
        logger.debug(
            "local_qdrant.create_collection", collection=collection, dim=vector_size
        )
        client.recreate_collection(
            collection_name=collection,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )
        return
    if current == vector_size:
        return
    logger.warning(
        "local_qdrant.recreate_collection_due_to_dim_mismatch",
        collection=collection,
        previous_dim=current,
        requested_dim=vector_size,
    )
    try:
        client.delete_collection(collection_name=collection)
    except Exception:
        logger.warning(
            "local_qdrant.delete_collection_failed", collection=collection, exc_info=True
        )
    client.recreate_collection(
        collection_name=collection,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )


def upsert(
    ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    payloads: Sequence[Mapping[str, Any]],
    *,
    collection: str | None = None,
) -> None:
    """Upsert embeddings into the configured collection."""

    if not ids:
        return
    if len(ids) != len(vectors) or len(ids) != len(payloads):
        msg = "ids, vectors, and payloads must have matching lengths"
        raise ValueError(msg)
    settings = get_settings()
    collection_name = collection or settings.qdrant_collection
    ensure_collection(collection_name, len(vectors[0]))
    client = get_client()
    points = [
        rest.PointStruct(
            id=str(idx),
            vector=list(vector),
            payload=dict(payload),
        )
        for idx, vector, payload in zip(ids, vectors, payloads, strict=True)
    ]
    client.upsert(collection_name=collection_name, points=points)


def search(
    query: str,
    *,
    top_k: int = 5,
    collection: str | None = None,
) -> list[dict[str, Any]]:
    """Search the local Qdrant store using embeddings for the query."""

    if not query.strip():
        return []
    settings = get_settings()
    collection_name = collection or settings.qdrant_collection
    embedder = get_embedder()
    vector = embedder.embed([query])[0]
    client = get_client()
    try:
        hits = client.search(
            collection_name=collection_name,
            query_vector=list(vector),
            limit=top_k,
        )
    except Exception:
        logger.warning("local_qdrant.search_failed", collection=collection_name, exc_info=True)
        return []
    results: list[dict[str, Any]] = []
    for point in hits:
        results.append(
            {
                "id": str(point.id),
                "score": float(point.score),
                "payload": dict(point.payload or {}),
            }
        )
    return results


def _extract_chunks(path: Path, *, book_id: str, size: int, overlap: int) -> list[Chunk]:
    pages = [
        (page["text"], page["page_num"])
        for page in extract_pages(path)
        if page["text"].strip()
    ]
    if not pages:
        return []
    return sliding_window_chunks(pages, book_id=book_id, size=size, overlap=overlap)


def index_path(
    path: str | Path,
    *,
    collection: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Index a PDF located at ``path`` into the local Qdrant collection."""

    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    book_id = pdf_path.stem
    chunks = _extract_chunks(pdf_path, book_id=book_id, size=size, overlap=overlap)
    if not chunks:
        logger.warning("local_qdrant.no_chunks_extracted", path=str(pdf_path))
        return []
    embedder = get_embedder()
    vectors = embedder.embed(chunk.text for chunk in chunks)
    if not vectors:
        logger.warning("local_qdrant.embedding_failed", path=str(pdf_path))
        return []
    vector_size = getattr(embedder, "dim", len(vectors[0]))
    ensure_collection(collection or settings.qdrant_collection, vector_size)
    payloads: list[dict[str, Any]] = []
    ids: list[str] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        ids.append(chunk.chunk_id)
        payloads.append(
            {
                "text": chunk.text,
                "book_id": chunk.book_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }
        )
    upsert(ids, vectors, payloads, collection=collection)
    return ids

