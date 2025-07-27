"""Index two sample text files into Qdrant."""

from pathlib import Path

from core.chunk.sliding import chunk_text
from core.embed.mock import MockEmbedder
from core.vector.qdrant import VectorStore


def main() -> None:
    samples = [Path("docs/sample1.txt"), Path("docs/sample2.txt")]
    embedder = MockEmbedder()
    store = VectorStore()
    for sample in samples:
        lines = sample.read_text().splitlines()
        chunks = chunk_text(lines)
        embeddings = embedder.embed(chunks)
        payloads = [{"text": c, "source": sample.name} for c in chunks]
        store.upsert(embeddings, payloads)


if __name__ == "__main__":
    main()
