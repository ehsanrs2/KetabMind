from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


class VectorStore:
    def __init__(
        self,
        *,
        mode: str,
        collection: str,
        vector_size: int,
        location: str | None = None,
        url: str | None = None,
    ) -> None:
        self.collection = collection
        if mode == "local":
            self.client = QdrantClient(path=location)
        else:
            assert url, "QDRANT_URL is required for remote mode"
            self.client = QdrantClient(url=url)
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=rest.VectorParams(size=self.vector_size, distance=rest.Distance.COSINE),
        )

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection)

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        norm_ids = self._normalize_ids(ids)
        points = rest.Batch(
            ids=norm_ids,
            vectors=vectors.tolist(),
            payloads=payloads,
        )
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, vector: np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=top_k,
        )
        hits: list[dict[str, Any]] = []
        for p in res:
            hits.append(
                {
                    "id": str(p.id),
                    "score": float(p.score),
                    "payload": dict(p.payload or {}),
                }
            )
        return hits

    def retrieve_existing(self, ids: list[str]) -> list[str]:
        if not ids:
            return []
        norm_ids = self._normalize_ids(ids)
        recs = self.client.retrieve(collection_name=self.collection, ids=norm_ids)
        return [str(r.id) for r in recs]

    def _normalize_ids(self, ids: list[str]) -> list[str]:
        out: list[str] = []
        for i in ids:
            try:
                uuid.UUID(i)
                out.append(i)
            except Exception:
                out.append(str(uuid.uuid5(uuid.NAMESPACE_URL, i)))
        return out
