from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import structlog
from qdrant_client.http import models as rest

from core.chunk.chunker import sliding_window_chunks
from core.config import settings
from core.embed import get_embedder
from core.ingest.pdf_to_text import Page, pdf_to_pages
from ingest.pipeline import pages_to_records, write_records
from core.vector.qdrant_client import VectorStore
from utils.hash import sha256_file

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


def _manifest_key(collection: str, file_hash: str) -> str:
    return f"{collection}:{file_hash}"


def _new_version() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("v%Y%m%d%H%M%S%f")


@dataclass
class IndexedFile:
    book_id: str
    version: str
    file_hash: str
    indexed_chunks: int


@dataclass
class IndexResult:
    new: int
    skipped: int
    collection: str
    book_id: str
    version: str
    file_hash: str
    indexed_chunks: int


def _read_text_file(path: Path) -> list[tuple[str, int]]:
    text = path.read_text(encoding="utf-8")
    return [(text, 1)]


def _clean_metadata(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {}
    return {
        str(key): value
        for key, value in meta.items()
        if value not in (None, "", [], {})
    }


def _jsonl_path(book_id: str) -> Path:
    return _manifest_dir() / f"{book_id}.jsonl"


def index_path(
    in_path: Path,
    collection: str | None = None,
    *,
    file_hash: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> IndexResult:
    """Index a file path into Qdrant.

    Returns counts of new and skipped chunks and the collection used.
    """
    if not in_path.exists():
        raise SystemExit(f"File not found: {in_path}")

    store_collection = collection or settings.qdrant_collection

    file_hash = file_hash or sha256_file(in_path)
    manifest = _load_manifest()
    manifest_key = _manifest_key(store_collection, file_hash)
    cached = manifest.get(manifest_key)
    if cached:
        book_id = str(cached.get("book_id"))
        version = str(cached.get("version"))
        indexed_chunks = int(cached.get("indexed_chunks", 0)) or int(
            cached.get("chunks", 0)
        )
        skipped_count = indexed_chunks or 1
        log.info(
            "indexed",
            book_id=book_id,
            version=version,
            new=0,
            skipped=skipped_count,
            total=skipped_count,
        )
        return IndexResult(
            new=0,
            skipped=skipped_count,
            collection=store_collection,
            book_id=book_id,
            version=version,
            file_hash=file_hash,
            indexed_chunks=skipped_count,
        )

    book_id = str(uuid.uuid4())
    version = _new_version()
    meta_payload = _clean_metadata(metadata)

    if in_path.suffix.lower() == ".pdf":
        pages = pdf_to_pages(in_path)
        texts_pages = [(p.text, p.page_num) for p in pages]
    elif in_path.suffix.lower() in {".txt", ".md"}:
        texts_pages = _read_text_file(in_path)
        pages = [Page(page_num=page, text=text) for text, page in texts_pages]
    else:
        raise SystemExit("Unsupported file type; use .pdf or .txt")

    records = pages_to_records(
        pages,
        book_id=book_id,
        version=version,
        file_hash=file_hash,
        metadata=meta_payload,
    )
    write_records(records, _jsonl_path(book_id))

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
    indexed_chunks = len(chunks)
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
                "file_hash": file_hash,
                "version": version,
                "indexed_chunks": indexed_chunks,
                "meta": dict(meta_payload),
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

    manifest_entry = {
        "book_id": book_id,
        "path": str(in_path),
        "file_hash": file_hash,
        "indexed_chunks": len(ids),
        "version": version,
    }
    if meta_payload:
        manifest_entry["meta"] = meta_payload
    manifest_entry["jsonl_path"] = str(_jsonl_path(book_id))
    manifest[manifest_key] = manifest_entry
    _save_manifest(manifest)

    log.info(
        "indexed",
        book_id=book_id,
        version=version,
        new=new_count,
        skipped=skipped_count,
        total=len(ids),
    )
    return IndexResult(
        new=new_count,
        skipped=skipped_count,
        collection=store_collection,
        book_id=book_id,
        version=version,
        file_hash=file_hash,
        indexed_chunks=len(ids),
    )


def find_indexed_file(collection: str, file_hash: str) -> IndexedFile | None:
    """Return metadata for an indexed file if it already exists."""

    manifest = _load_manifest()
    cached = manifest.get(_manifest_key(collection, file_hash))
    if cached:
        return IndexedFile(
            book_id=str(cached.get("book_id")),
            version=str(cached.get("version")),
            file_hash=file_hash,
            indexed_chunks=int(cached.get("indexed_chunks", 0))
            or int(cached.get("chunks", 0)),
        )

    with VectorStore(
        mode=settings.qdrant_mode,
        location=settings.qdrant_location,
        url=settings.qdrant_url,
        collection=collection,
        vector_size=1,
    ) as store:
        try:
            existing, _ = store.client.scroll(
                collection_name=collection,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="file_hash",
                            match=rest.MatchValue(value=file_hash),
                        )
                    ]
                ),
                limit=1,
            )
        except ValueError as exc:
            if "not found" in str(exc).lower():
                return None
            raise

    if not existing:
        return None

    payload = dict(existing[0].payload or {})
    book_id = str(payload.get("book_id", ""))
    version = str(payload.get("version", "")) or _new_version()
    indexed_chunks = int(payload.get("indexed_chunks") or 0)
    return IndexedFile(
        book_id=book_id,
        version=version,
        file_hash=file_hash,
        indexed_chunks=indexed_chunks,
    )
