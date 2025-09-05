"""Retrieve documents using the vector store."""

from ..embed.base import Embedder
from ..vector import qdrant
from ..vector.qdrant import VectorStore


class Retriever:
    """k-NN retriever with optional filters."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[qdrant.ChunkPayload]:
        embedding = self.embedder.embed([query])[0]
        return self.store.query(embedding, top_k)
