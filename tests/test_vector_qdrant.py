from __future__ import annotations

import tempfile

import numpy as np

from core.vector.qdrant_client import VectorStore


def test_vectorstore_upsert_query_local() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        vs = VectorStore(mode="local", location=tmp, url=None, collection="test_col", vector_size=8)
        vs.ensure_collection()
        ids = ["a", "b"]
        vecs = np.eye(2, 8, dtype=np.float32)
        payloads = [
            {"text": "alpha", "book_id": "b", "page_start": 1, "page_end": 1, "chunk_id": "c1"},
            {"text": "beta", "book_id": "b", "page_start": 2, "page_end": 2, "chunk_id": "c2"},
        ]
        vs.upsert(ids, vecs, payloads)
        res = vs.query(vecs[0], top_k=1)
        assert res and res[0]["payload"]["text"] == "alpha"
