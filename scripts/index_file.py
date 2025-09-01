from __future__ import annotations

import uuid
from pathlib import Path

from core.chunk.chunker import sliding_window_chunks
from core.config import settings
from core.embed import get_embedder
from core.ingest.pdf_to_text import pdf_to_pages
from core.vector.qdrant_client import VectorStore


def _book_id_from_path(path: Path) -> str:
    return path.stem


def _read_text_file(path: Path) -> list[tuple[str, int]]:
    text = path.read_text(encoding="utf-8")
    return [(text, 1)]


def index_path(in_path: Path) -> None:
    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")

    if in_path.suffix.lower() == ".pdf":
        pages = pdf_to_pages(in_path)
        texts_pages = [(p.text, p.page_num) for p in pages]
    elif in_path.suffix.lower() in {".txt", ".md"}:
        texts_pages = _read_text_file(in_path)
    else:
        raise SystemExit("Unsupported file type; use .pdf or .txt")

    book_id = _book_id_from_path(in_path)

    embedder = get_embedder()
    store = VectorStore(
        mode=settings.qdrant_mode,
        location=settings.qdrant_location,
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        vector_size=getattr(embedder, "dim", len(embedder.embed(["test"])[0])),
    )
    store.ensure_collection()

    chunks = sliding_window_chunks(
        texts_pages,
        book_id=book_id,
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    ids: list[str] = []
    payloads: list[dict] = []
    for ch in chunks:
        # Use deterministic UUIDv5 for Qdrant-compatible point IDs
        name = f"{book_id}||{ch.text}"
        uid = str(uuid.uuid5(uuid.NAMESPACE_URL, name))
        ids.append(uid)
        payloads.append(
            {
                "text": ch.text,
                "book_id": ch.book_id,
                "page_start": ch.page_start,
                "page_end": ch.page_end,
                "chunk_id": ch.chunk_id,
            }
        )

    # Determine which are new
    existing = set(store.retrieve_existing(ids))
    new_mask = [pid not in existing for pid in ids]
    new_count = sum(1 for m in new_mask if m)
    skipped_count = len(ids) - new_count

    import numpy as np

    vecs = np.asarray(embedder.embed([p["text"] for p in payloads]), dtype=np.float32)
    store.upsert(ids=ids, vectors=vecs, payloads=payloads)

    print(f"Indexed book={book_id} new={new_count} skipped={skipped_count} total={len(ids)}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Input file (.pdf or .txt)")
    args = parser.parse_args()
    in_path = Path(args.path)
    index_path(in_path)


if __name__ == "__main__":
    main()
