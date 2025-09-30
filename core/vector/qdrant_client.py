from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import structlog

FloatArray: TypeAlias = NDArray[np.floating[Any]]

log = structlog.get_logger(__name__)


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
        vector_size: int | None = None,
        embedding_adapter: Any | None = None,
        location: str | None = None,
        url: str | None = None,
    ) -> None:
        self.collection = collection
        if mode == "local":
            self.client = QdrantClient(path=location)
        else:
            assert url, "QDRANT_URL is required for remote mode"
            self.client = QdrantClient(url=url)
        if embedding_adapter is not None:
            adapter_dim = getattr(embedding_adapter, "dim", None)
            if adapter_dim is None:
                msg = "embedding_adapter must expose a 'dim' attribute"
                raise AttributeError(msg)
            self.vector_size = int(adapter_dim)
        elif vector_size is not None:
            self.vector_size = int(vector_size)
        else:
            msg = "vector_size or embedding_adapter is required"
            raise ValueError(msg)
        self.ensure_collection()

    def __enter__(self) -> "VectorStore":  # pragma: no cover - simple context helper
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - simple context helper
        self.close()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def ensure_collection(self) -> None:
        self.recreate_collection_if_needed(self.collection, self.vector_size)

    def recreate_collection_if_needed(self, name: str, dim: int) -> None:
        current = self._current_vector_size(name)
        if current is None:
            self.client.recreate_collection(
                collection_name=name,
                vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
            )
            return
        if current == dim:
            return
        log.warning(
            "recreating_qdrant_collection_due_to_dim_mismatch",
            collection=name,
            previous_dim=current,
            requested_dim=dim,
        )
        try:
            self.client.delete_collection(collection_name=name)
        except Exception:
            log.warning(
                "failed_deleting_qdrant_collection_before_recreate",
                collection=name,
                exc_info=True,
            )
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
        )

    def delete_collection(self) -> None:
        self.client.delete_collection(collection_name=self.collection)

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
        payload_list: list[dict[str, Any]] = []
        for payload in payloads:
            payload_dict = dict(payload)
            meta = payload_dict.get("meta")
            if isinstance(meta, Mapping):
                payload_dict["meta"] = dict(meta)
            else:
                payload_dict["meta"] = {}
            payload_list.append(payload_dict)
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

    def _current_vector_size(self, collection_name: str | None = None) -> int | None:
        try:
            info = self.client.get_collection(collection_name or self.collection)
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
