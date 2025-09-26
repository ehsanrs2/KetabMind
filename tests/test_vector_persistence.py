import importlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from qdrant_client.http import models as rest

import core.config as cfg
import core.vector.qdrant as qdrant

F = TypeVar("F", bound=Callable[..., Any])
integration = cast(Callable[[F], F], pytest.mark.integration)


@integration
def test_vector_persistence(tmp_path: Path) -> None:
    os.environ["QDRANT_MODE"] = "local"
    os.environ["QDRANT_LOCATION"] = str(tmp_path)
    importlib.reload(cfg)
    importlib.reload(qdrant)
    store1 = qdrant.VectorStore()
    store1.collection = "books"
    vec = [0.0] * 384
    payload = cast(
        qdrant.ChunkPayload,
        {
            "text": "t",
            "book_id": "b",
            "chapter": None,
            "page_start": 1,
            "page_end": 1,
            "chunk_id": "c1",
            "content_hash": "h",
        },
    )
    store1.upsert([vec], [payload])
    store1.client.close()
    store2 = qdrant.VectorStore()
    info = store2.client.get_collection("books")
    params = info.config.params
    size = 0
    if isinstance(params, rest.CollectionParams):
        vectors = params.vectors
        if isinstance(vectors, rest.VectorParams):
            size = int(vectors.size)
        elif isinstance(vectors, dict):
            size = int(vectors.get("size", 0))
    assert size == 384
