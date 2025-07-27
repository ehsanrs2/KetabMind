"""Qdrant vector store client."""

from typing import Iterable, List, Optional, TypedDict

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from ..config import settings
from typing import cast


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
        if settings.qdrant_mode == "local" or not settings.qdrant_url:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection

    def upsert(
        self, embeddings: Iterable[List[float]], payloads: Iterable[ChunkPayload]
    ) -> None:
        points = [
            rest.PointStruct(id=i, vector=vec, payload=dict(payload))
            for i, (vec, payload) in enumerate(zip(embeddings, payloads))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, embedding: List[float], top_k: int) -> List[ChunkPayload]:
        result = self.client.search(
            collection_name=self.collection, query_vector=embedding, limit=top_k
        )
        return [cast(ChunkPayload, hit.payload) for hit in result]
