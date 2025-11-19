"""Qdrant vector store client."""
import uuid
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
    return QdrantClient(url=settings.qdrant_url)

class ChunkPayload(TypedDict):
    text: str
    book_id: str
    chapter: str | None
    page_start: int
    page_end: int
    chunk_id: str
    content_hash: str

class VectorStore:
    def __init__(self) -> None:
        self.client = make_qdrant_client()
        self.collection = settings.qdrant_collection

    def ensure_collection(self, dim: int) -> None:
        current = self._current_vector_size()
        if current is None:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
            )
            return
        if current != dim:
            log.warning("recreating_qdrant_collection_dim_mismatch", current=current, new=dim)
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
        if isinstance(params, rest.CollectionParams) and isinstance(params.vectors, rest.VectorParams):
            return int(params.vectors.size)
        return None

    def upsert(self, embeddings: Iterable[list[float]], payloads: Iterable[ChunkPayload]) -> tuple[int, int]:
        new_points = []
        skipped = 0
        ensured = False
        
        for vec, payload in zip(embeddings, payloads, strict=False):
            if not ensured:
                self.ensure_collection(len(vec))
                ensured = True
            
            # استفاده از UUID برای جلوگیری از تداخل و ایجاد شناسه یکتا
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, payload["content_hash"] + payload["book_id"]))
            
            new_points.append(rest.PointStruct(id=point_id, vector=vec, payload=dict(payload)))

        if new_points:
            self.client.upsert(collection_name=self.collection, points=new_points)
            
        return len(new_points), skipped

    def query(self, embedding: list[float], top_k: int, book_id: str | None = None) -> list[ChunkPayload]:
        self.ensure_collection(len(embedding))
        
        query_filter = None
        if book_id:
            query_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="book_id",
                        match=rest.MatchValue(value=book_id),
                    )
                ]
            )

        result = self.client.search(
            collection_name=self.collection, 
            query_vector=embedding, 
            limit=top_k,
            query_filter=query_filter
        )
        return [cast(ChunkPayload, hit.payload) for hit in result]

    def delete_by_book_id(self, book_id: str) -> None:
        """Delete all vectors for a specific book."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=rest.FilterSelector(
                filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="book_id",
                            match=rest.MatchValue(value=book_id),
                        )
                    ]
                )
            ),
        )

    def wipe_collection(self) -> None:
        with suppress(Exception):
            self.client.delete_collection(collection_name=self.collection)