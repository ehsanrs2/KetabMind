"""Qdrant vector store client."""

from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import TypedDict, cast

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from ..config import settings

log = structlog.get_logger(__name__)


def make_qdrant_client() -> QdrantClient:
    """Build Qdrant client based on settings."""
    if settings.qdrant_mode == "local":
        location = settings.qdrant_location
        if location == ":memory:":
            return QdrantClient(path=location)
        path = Path(location)
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    if settings.qdrant_mode == "docker":
        if not settings.qdrant_url:
            raise RuntimeError("QDRANT_URL required for docker mode")
        return QdrantClient(url=settings.qdrant_url)
    if not settings.qdrant_url:
        raise RuntimeError("QDRANT_URL required for remote mode")
    return QdrantClient(url=settings.qdrant_url)


class ChunkPayload(TypedDict):
    """Payload stored in Qdrant."""

    text: str
    book_id: str
    chapter: str | None
    page_start: int
    page_end: int
    chunk_id: str
    content_hash: str


class VectorStore:
    """Simple Qdrant wrapper."""

    def __init__(self) -> None:
        self.client = make_qdrant_client()
        self.collection = settings.qdrant_collection

    def ensure_collection(self, dim: int) -> None:
        """Ensure the backing collection exists with the expected vector size."""
        current = self._current_vector_size()
        if current is None:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
            )
            return
        if current != dim:
            log.warning(
                "recreating_qdrant_collection_due_to_dim_mismatch",
                collection=self.collection,
                previous_dim=current,
                requested_dim=dim,
            )
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
            )

    def _current_vector_size(self) -> int | None:
        try:
            info = self.client.get_collection(self.collection)
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

    def upsert(
        self, embeddings: Iterable[list[float]], payloads: Iterable[ChunkPayload]
    ) -> tuple[int, int]:
        new_vecs: list[list[float]] = []
        new_payloads: list[ChunkPayload] = []
        skipped = 0
        ensured = False
        for vec, payload in zip(embeddings, payloads, strict=False):
            if not ensured:
                self.ensure_collection(len(vec))
                ensured = True
            existing, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="content_hash",
                            match=rest.MatchValue(value=payload["content_hash"]),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
            )
            if existing:
                skipped += 1
                continue
            new_vecs.append(vec)
            new_payloads.append(payload)

        points = [
            rest.PointStruct(id=i, vector=vec, payload=dict(pl))
            for i, (vec, pl) in enumerate(zip(new_vecs, new_payloads, strict=False))
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return len(points), skipped

    def query(self, embedding: list[float], top_k: int) -> list[ChunkPayload]:
        self.ensure_collection(len(embedding))
        result = self.client.search(
            collection_name=self.collection, query_vector=embedding, limit=top_k
        )
        return [cast(ChunkPayload, hit.payload) for hit in result]

    def wipe_collection(self) -> None:
        """Drop the current collection."""
        with suppress(Exception):
            self.client.delete_collection(collection_name=self.collection)
