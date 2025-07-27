"""Qdrant vector store client."""

from typing import Any, Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from ..config import settings


class VectorStore:
    """Simple Qdrant wrapper."""

    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection

    def upsert(
        self, embeddings: Iterable[List[float]], payloads: Iterable[dict[str, Any]]
    ) -> None:
        points = [
            rest.PointStruct(id=i, vector=vec, payload=payload)
            for i, (vec, payload) in enumerate(zip(embeddings, payloads))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, embedding: List[float], top_k: int) -> List[dict[str, Any]]:
        result = self.client.search(
            collection_name=self.collection, query_vector=embedding, limit=top_k
        )
        return [hit.payload for hit in result]
