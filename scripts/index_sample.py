"""Index two sample text files into Qdrant."""

from pathlib import Path

from core.chunk.sliding import chunk_text
from core.embed.mock import MockEmbedder
from core.vector.qdrant import VectorStore, ChunkPayload
from typing import cast


def main() -> None:
    samples = [Path("docs/sample1.txt"), Path("docs/sample2.txt")]
    embedder = MockEmbedder()
    store = VectorStore()
    for sample in samples:
        lines = sample.read_text().splitlines()
        chunks = chunk_text(lines)
        embeddings = embedder.embed(chunks)
        payloads = cast(
            list[ChunkPayload],
            [
                {
                    "text": c,
                    "book_id": sample.name,
                    "chapter": None,
                    "page_start": 0,
                    "page_end": 0,
                    "chunk_id": f"{sample.stem}-{i}",
                    "content_hash": "0",
                }
                for i, c in enumerate(chunks)
            ],
        )
        store.upsert(embeddings, payloads)


if __name__ == "__main__":
    main()
