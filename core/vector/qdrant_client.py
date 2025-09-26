from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

FloatArray: TypeAlias = NDArray[np.floating[Any]]


@runtime_checkable
class _SupportsToList(Protocol):
    def tolist(self) -> list[Any]:
        """Return a Python representation of the array."""


MatrixInput: TypeAlias = FloatArray | Sequence[Sequence[float]] | _SupportsToList
VectorInput: TypeAlias = FloatArray | Sequence[float] | _SupportsToList


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

    def upsert(
        self,
        ids: Iterable[str],
        vectors: MatrixInput,
        payloads: Iterable[Mapping[str, Any]],
    ) -> None:
        id_list = list(ids)
        if not id_list:
            return
        norm_ids = self._normalize_ids(id_list)
        vector_list = self._to_list(vectors)
        if len(vector_list) != len(id_list):  # pragma: no cover - defensive
            msg = "vectors length must match ids length"
            raise ValueError(msg)
        payload_list = [dict(p) for p in payloads]
        if len(payload_list) != len(id_list):  # pragma: no cover - defensive
            msg = "payloads length must match ids length"
            raise ValueError(msg)
        points = rest.Batch(ids=norm_ids, vectors=vector_list, payloads=payload_list)
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, vector: VectorInput, top_k: int = 3) -> list[dict[str, Any]]:
        query_vector = self._to_list(vector)
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
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

    def retrieve_existing(self, ids: Iterable[str]) -> list[str]:
        id_list = list(ids)
        if not id_list:
            return []
        norm_ids = self._normalize_ids(id_list)
        recs = self.client.retrieve(collection_name=self.collection, ids=norm_ids)
        return [str(r.id) for r in recs]

    def _normalize_ids(self, ids: Iterable[str]) -> list[str]:
        out: list[str] = []
        for i in ids:
            try:
                uuid.UUID(i)
                out.append(i)
            except Exception:
                out.append(str(uuid.uuid5(uuid.NAMESPACE_URL, i)))
        return out

    def _to_list(self, value: MatrixInput | VectorInput) -> list[Any]:
        if isinstance(value, _SupportsToList):
            return value.tolist()
        items: list[Any] = list(value)
        return [self._ensure_inner(v) for v in items]

    def _ensure_inner(self, item: Any) -> Any:
        if isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray):
            return [self._ensure_inner(v) for v in item]
        if isinstance(item, _SupportsToList):
            return item.tolist()
        return item
