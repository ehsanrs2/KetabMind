from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Sequence

from .http import models as rest

__all__ = ["QdrantClient"]


@dataclass
class _Record:
    id: Any
    payload: dict[str, Any]
    score: float = 1.0


class QdrantClient:
    def __init__(self, *, path: str | None = None, url: str | None = None) -> None:
        self.path = path
        self.url = url
        self._collections: Dict[str, Dict[str, Any]] = {}

    def recreate_collection(self, collection_name: str, vectors_config: rest.VectorParams) -> None:
        self._collections[collection_name] = {
            "config": rest.CollectionParams(vectors=vectors_config),
            "points": {},
        }

    def delete_collection(self, collection_name: str) -> None:
        self._collections.pop(collection_name, None)

    def get_collection(self, collection_name: str) -> Any:
        if collection_name not in self._collections:
            raise RuntimeError("collection does not exist")
        config = self._collections[collection_name]["config"]
        return SimpleNamespace(config=SimpleNamespace(params=config))

    def upsert(self, collection_name: str, points: Any) -> None:
        collection = self._collections.setdefault(
            collection_name,
            {"config": rest.CollectionParams(vectors=rest.VectorParams(size=0, distance=rest.Distance.COSINE)), "points": {}},
        )
        store = collection["points"]
        if isinstance(points, list):
            items = ((p.id, p.vector, p.payload) for p in points)
        else:
            items = zip(points.ids, points.vectors, points.payloads, strict=False)
        for idx, vector, payload in items:
            store[str(idx)] = {"vector": list(vector), "payload": dict(payload)}

    def scroll(
        self,
        collection_name: str,
        *,
        scroll_filter: rest.Filter,
        limit: int,
        **_: Any,
    ) -> tuple[list[_Record], None]:
        collection = self._collections.get(collection_name)
        if not collection:
            return ([], None)
        matches: list[_Record] = []
        for point_id, point in collection["points"].items():
            payload = point["payload"]
            if _match_filter(payload, scroll_filter):
                matches.append(_Record(point_id, dict(payload)))
            if len(matches) >= limit:
                break
        return (matches, None)

    def search(self, collection_name: str, query_vector: Sequence[float], limit: int) -> list[_Record]:
        collection = self._collections.get(collection_name)
        if not collection:
            return []
        records: list[_Record] = []
        for point_id, point in list(collection["points"].items())[:limit]:
            payload = dict(point["payload"])
            score = _cosine_similarity(query_vector, point["vector"])
            records.append(_Record(point_id, payload, score=score))
        return records

    def retrieve(self, collection_name: str, ids: Iterable[str]) -> list[_Record]:
        collection = self._collections.get(collection_name)
        if not collection:
            return []
        store = collection["points"]
        return [_Record(i, dict(store[i]["payload"])) for i in ids if i in store]

    def close(self) -> None:  # pragma: no cover - nothing to close in stub
        return None


def _match_filter(payload: Mapping[str, Any], flt: rest.Filter) -> bool:
    for condition in flt.must:
        expected = condition.match.value
        if payload.get(condition.key) != expected:
            return False
    return True


def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2, strict=False))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
