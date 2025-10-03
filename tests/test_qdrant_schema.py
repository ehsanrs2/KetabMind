from __future__ import annotations

import tempfile
from types import SimpleNamespace

import pytest

qdrant = pytest.importorskip("qdrant_client")
rest = pytest.importorskip("qdrant_client.http.models")
QdrantClient = qdrant.QdrantClient

from core.vector.qdrant_client import VectorStore  # noqa: E402


def _vector_size(info: rest.CollectionInfo) -> int:
    params = getattr(getattr(info, "config", None), "params", None)
    if isinstance(params, rest.CollectionParams):
        vectors = params.vectors
        if isinstance(vectors, rest.VectorParams):
            return int(vectors.size)
        if isinstance(vectors, dict) and "size" in vectors:
            return int(vectors["size"])
    raise AssertionError("Unable to determine vector size")


def test_vectorstore_recreates_mismatched_collection() -> None:
    adapter = SimpleNamespace(dim=384)
    collection = "test_schema"
    with tempfile.TemporaryDirectory() as tmp:
        client = QdrantClient(path=tmp)
        client.recreate_collection(
            collection_name=collection,
            vectors_config=rest.VectorParams(size=512, distance=rest.Distance.COSINE),
        )

        store = VectorStore(
            mode="local",
            location=tmp,
            url=None,
            collection=collection,
            embedding_adapter=adapter,
        )

        info = store.client.get_collection(collection)
        assert _vector_size(info) == adapter.dim
