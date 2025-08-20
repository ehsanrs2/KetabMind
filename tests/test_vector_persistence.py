import importlib
import os
from pathlib import Path
from typing import cast

import pytest

import core.config as cfg
import core.vector.qdrant as qdrant


@pytest.mark.integration  # type: ignore[misc]
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
    assert info.config.params.vectors.size == 384
