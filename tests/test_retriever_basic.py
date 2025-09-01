import hashlib
import importlib

import pytest


@pytest.mark.not_slow  # type: ignore[misc]
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
