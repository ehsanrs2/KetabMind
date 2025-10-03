from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Distance(Enum):
    COSINE = "Cosine"


@dataclass
class VectorParams:
    size: int
    distance: Distance


@dataclass
class CollectionParams:
    vectors: VectorParams | dict[str, Any]


@dataclass
class MatchValue:
    value: Any


@dataclass
class FieldCondition:
    key: str
    match: MatchValue


@dataclass
class Filter:
    must: Sequence[FieldCondition]


@dataclass
class PointStruct:
    id: Any
    vector: Sequence[float]
    payload: dict[str, Any]


@dataclass
class Batch:
    ids: Sequence[Any]
    vectors: Sequence[Sequence[float]]
    payloads: Sequence[dict[str, Any]]
