from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable
from typing import Any, TypeVar, cast

import pytest

F = TypeVar("F", bound=Callable[..., Any])
not_slow = cast(Callable[[F], F], pytest.mark.not_slow)


@not_slow
def test_basic_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_MODEL", "mock")
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", ":memory:")
    import core.config as config
    import core.vector.qdrant as qdrant_module

    importlib.reload(config)
    importlib.reload(qdrant_module)

    from core.embed import get_embedder
    from core.retrieve.retriever import Retriever

    VectorStore = qdrant_module.VectorStore

    store = VectorStore()
    embedder = get_embedder()
    chunks = ["The sky is blue", "Cats are cute", "Python programming language"]
    embeddings = embedder.embed(chunks)
    payloads: list[qdrant_module.ChunkPayload] = []
    for i, text in enumerate(chunks):
        payloads.append(
            {
                "text": text,
                "book_id": "test",
                "chapter": None,
                "page_start": 0,
                "page_end": 0,
                "chunk_id": str(i),
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    store.upsert(embeddings, payloads)
    retriever = Retriever()
    retriever.store = store
    results = retriever.retrieve("blue sky", top_k=3)
    assert "sky" in results[0].text.lower()


@not_slow
def test_vector_search_filters_by_book_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_MODEL", "mock")
    monkeypatch.setenv("QDRANT_MODE", "local")
    monkeypatch.setenv("QDRANT_LOCATION", ":memory:")

    import core.config as config
    import core.vector.qdrant as qdrant_module

    importlib.reload(config)
    importlib.reload(qdrant_module)

    from core.embed import get_embedder
    from core.retrieve.retriever import Retriever

    VectorStore = qdrant_module.VectorStore

    store = VectorStore()
    embedder = get_embedder()
    chunks = [
        ("book-a", "Transformers revolve around attention mechanisms."),
        ("book-b", "Economic indicators track inflation and growth."),
    ]
    embeddings = embedder.embed([text for _, text in chunks])
    payloads: list[qdrant_module.ChunkPayload] = []
    for i, (book_id, text) in enumerate(chunks):
        payloads.append(
            {
                "text": text,
                "book_id": book_id,
                "chapter": None,
                "page_start": i,
                "page_end": i,
                "chunk_id": f"{book_id}-{i}",
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    store.upsert(embeddings, payloads)

    retriever = Retriever()
    retriever.store = store

    results = retriever.retrieve(
        "What do transformers emphasize?", top_k=2, book_id="book-a"
    )

    assert results, "expected results for the requested book"
    assert all(chunk.book_id == "book-a" for chunk in results)
