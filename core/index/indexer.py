from __future__ import annotations

import json
import uuid
import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from core.chunk.chunker import sliding_window_chunks
from core.config import settings
from core.embed import get_embedder
from core.ingest.pdf_to_text import pdf_to_pages
from core.vector.qdrant_client import VectorStore

log = structlog.get_logger()

_MANIFEST_FILE = ".indexed_files.json"


def _manifest_dir() -> Path:
    location = settings.qdrant_location
    if not location or location == ":memory:":
        base = Path(tempfile.gettempdir()) / "ketabmind"
    else:
        base = Path(location)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _manifest_path() -> Path:
    return _manifest_dir() / _MANIFEST_FILE


def _load_manifest() -> dict[str, Any]:
    path = _manifest_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:  # pragma: no cover - defensive
        return {}


def _save_manifest(data: dict[str, Any]) -> None:
    path = _manifest_path()
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _book_id_from_path(path: Path) -> str:
    return path.stem


def _read_text_file(path: Path) -> list[tuple[str, int]]:
    text = path.read_text(encoding="utf-8")
    return [(text, 1)]


def index_path(in_path: Path, collection: str | None = None) -> tuple[int, int, str]:
    """Index a file path into Qdrant.

    Returns counts of new and skipped chunks and the collection used.
    """
    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")

    book_id = _book_id_from_path(in_path)
    store_collection = collection or settings.qdrant_collection

    file_hash = _hash_file(in_path)
    manifest = _load_manifest()
    manifest_key = f"{store_collection}:{file_hash}"
    cached = manifest.get(manifest_key)
    if cached:
        skipped_count = int(cached.get("chunks") or cached.get("new") or 0) or 1
        log.info(
            "indexed",
            book_id=book_id,
            new=0,
            skipped=skipped_count,
            total=skipped_count,
        )
        return 0, skipped_count, store_collection

    if in_path.suffix.lower() == ".pdf":
        pages = pdf_to_pages(in_path)
        texts_pages = [(p.text, p.page_num) for p in pages]
    elif in_path.suffix.lower() in {".txt", ".md"}:
        texts_pages = _read_text_file(in_path)
    else:
        raise SystemExit("Unsupported file type; use .pdf or .txt")

    embedder = get_embedder()
    vector_size = getattr(embedder, "dim", len(embedder.embed(["test"])[0]))

    chunks = sliding_window_chunks(
        texts_pages,
        book_id=book_id,
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    ids: list[str] = []
    payloads: list[dict[str, Any]] = []
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

    with VectorStore(
        mode=settings.qdrant_mode,
        location=settings.qdrant_location,
        url=settings.qdrant_url,
        collection=store_collection,
        vector_size=vector_size,
    ) as store:
        store.ensure_collection()
        existing = set(store.retrieve_existing(ids))
        new_mask = [pid not in existing for pid in ids]
        new_count = sum(1 for m in new_mask if m)
        skipped_count = len(ids) - new_count

        texts = [str(p["text"]) for p in payloads]
        vecs = np.asarray(embedder.embed(texts), dtype=np.float32)
        store.upsert(ids=ids, vectors=vecs, payloads=payloads)

    manifest[manifest_key] = {
        "book_id": book_id,
        "path": str(in_path),
        "chunks": len(ids),
        "new": new_count,
    }
    _save_manifest(manifest)

    log.info(
        "indexed",
        book_id=book_id,
        new=new_count,
        skipped=skipped_count,
        total=len(ids),
    )
    return new_count, skipped_count, store_collection
