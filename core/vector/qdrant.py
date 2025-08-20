"""Qdrant vector store client."""

from pathlib import Path
from typing import Iterable, List, Optional, TypedDict, cast

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from ..config import settings


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
    chapter: Optional[str]
    page_start: int
    page_end: int
    chunk_id: str
    content_hash: str


class VectorStore:
    """Simple Qdrant wrapper."""

    def __init__(self) -> None:
        self.client = make_qdrant_client()
        self.collection = settings.qdrant_collection

    def _ensure_collection(self, dim: int) -> None:
        """Create collection if missing."""
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=rest.VectorParams(
                    size=dim, distance=rest.Distance.COSINE
                ),
            )

    def upsert(
        self, embeddings: Iterable[List[float]], payloads: Iterable[ChunkPayload]
    ) -> tuple[int, int]:
        new_vecs: list[List[float]] = []
        new_payloads: list[ChunkPayload] = []
        skipped = 0
        ensured = False
        for vec, payload in zip(embeddings, payloads):
            if not ensured:
                self._ensure_collection(len(vec))
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
            )
            if existing:
                skipped += 1
                continue
            new_vecs.append(vec)
            new_payloads.append(payload)

        points = [
            rest.PointStruct(id=i, vector=vec, payload=dict(pl))
            for i, (vec, pl) in enumerate(zip(new_vecs, new_payloads))
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        return len(points), skipped

    def query(self, embedding: List[float], top_k: int) -> List[ChunkPayload]:
        result = self.client.search(
            collection_name=self.collection, query_vector=embedding, limit=top_k
        )
        return [cast(ChunkPayload, hit.payload) for hit in result]

    def wipe_collection(self) -> None:
        """Drop the current collection."""
        try:
            self.client.delete_collection(collection_name=self.collection)
        except Exception:
            pass
