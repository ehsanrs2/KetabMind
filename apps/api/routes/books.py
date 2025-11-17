"""Router exposing book management endpoints."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from types import SimpleNamespace

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from apps.api.auth import get_current_user
from apps.api.db import models, repositories
from apps.api.db.session import session_scope
from apps.api.schemas import BookRenameRequest
from apps.api.services.books import (
    delete_book_files,
    ensure_book_record,
    list_owner_book_ids,
)
from apps.api.services.users import ensure_owner
from core.config import settings
from core.index import (
    list_indexed_files,
    remove_indexed_book,
    update_indexed_book_metadata,
)
from core.vector.qdrant_client import VectorStore

router = APIRouter(prefix="/books", tags=["books"])

log = structlog.get_logger(__name__)


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _build_index_map(entries: list[Any]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for entry in entries:
        book_id = str(getattr(entry, "book_id", "") or "")
        if not book_id:
            continue
        existing = latest.get(book_id)
        candidate_version = getattr(entry, "version", None)
        if existing is None:
            latest[book_id] = entry
            continue
        current_version = getattr(existing, "version", None)
        if candidate_version and (not current_version or candidate_version > current_version):
            latest[book_id] = entry
    return latest


def _serialize_book(record: models.Book, index_entry: Any | None) -> dict[str, Any]:
    metadata = getattr(index_entry, "metadata", None) if index_entry is not None else None
    if isinstance(metadata, Mapping):
        metadata_payload = dict(metadata)
    else:
        metadata_payload = None
    status_value = "indexed" if index_entry is not None else "pending"
    return {
        "id": record.vector_id,
        "db_id": record.id,
        "vector_id": record.vector_id,
        "is_indexed": index_entry is not None or bool(record.vector_id),
        "title": record.title,
        "description": record.description,
        "created_at": _isoformat(record.created_at),
        "updated_at": _isoformat(record.updated_at),
        "status": status_value,
        "version": getattr(index_entry, "version", None) if index_entry is not None else None,
        "file_hash": getattr(index_entry, "file_hash", None) if index_entry is not None else None,
        "indexed_chunks": getattr(index_entry, "indexed_chunks", None)
        if index_entry is not None
        else None,
        "metadata": metadata_payload,
    }


def _vector_store() -> VectorStore:
    return VectorStore(
        mode=settings.qdrant_mode,
        location=settings.qdrant_location,
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        ensure_collection=False,
        vector_size=None,
    )


def _paginate(items: list[Any], offset: int, limit: int) -> list[Any]:
    return items[offset : offset + limit]


@router.get("")
def list_books(
    current_user: dict[str, Any] = Depends(get_current_user),
    *,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    owner_external_id = str(current_user.get("id") or "")
    if not owner_external_id:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid user profile")

    try:
        indexed_files = list_indexed_files(settings.qdrant_collection)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("books.list_index_failed", error=str(exc), exc_info=True)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Vector store unavailable")

    index_map = _build_index_map(list(indexed_files))

    with session_scope() as db_session:
        owner = ensure_owner(db_session, current_user)
        repo = repositories.BookRepository(db_session, owner.id)
        known_ids = {record.vector_id for record in repo.list() if record.vector_id}
        for vector_id in list_owner_book_ids(owner_external_id):
            if vector_id not in known_ids:
                ensure_book_record(
                    db_session,
                    owner,
                    book_id=vector_id,
                    metadata=getattr(index_map.get(vector_id), "metadata", None),
                    title_fallback=vector_id,
                )
        records = repo.list()
        total = len(records)
        paginated = _paginate(records, offset, limit)
        payload = [_serialize_book(record, index_map.get(record.vector_id or "")) for record in paginated]
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "books": payload,
        }


@router.get("/{book_id}")
def get_book(
    book_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    owner_external_id = str(current_user.get("id") or "")
    if not owner_external_id:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid user profile")

    try:
        indexed_files = list_indexed_files(settings.qdrant_collection)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("books.detail_index_failed", error=str(exc), exc_info=True)
        indexed_files = []

    index_map = _build_index_map(list(indexed_files))
    with session_scope() as db_session:
        owner = ensure_owner(db_session, current_user)
        repo = repositories.BookRepository(db_session, owner.id)
        record = repo.get_by_vector_id(book_id)
        if record is None:
            directory_ids = list_owner_book_ids(owner_external_id)
            if book_id in directory_ids:
                metadata = getattr(index_map.get(book_id), "metadata", None)
                record = ensure_book_record(
                    db_session,
                    owner,
                    book_id=book_id,
                    metadata=metadata,
                    title_fallback=book_id,
                )
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Book not found")
        return {"book": _serialize_book(record, index_map.get(book_id))}


@router.patch("/{book_id}/rename")
def rename_book(
    book_id: str,
    payload: BookRenameRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    new_title = (payload.title or "").strip()
    if not new_title:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Title is required")

    description = (payload.description or "").strip() or None

    try:
        indexed_files = list_indexed_files(settings.qdrant_collection)
    except Exception as exc:  # pragma: no cover - defensive logging
        log.warning("books.rename_index_failed", error=str(exc), exc_info=True)
        indexed_files = []

    index_map = _build_index_map(list(indexed_files))

    record_snapshot: dict[str, Any]
    with session_scope() as db_session:
        owner = ensure_owner(db_session, current_user)
        repo = repositories.BookRepository(db_session, owner.id)
        record = repo.rename(book_id, title=new_title, description=description)
        if record is None:
            record = ensure_book_record(
                db_session,
                owner,
                book_id=book_id,
                metadata=getattr(index_map.get(book_id), "metadata", None),
                title_fallback=new_title,
            )
            record.title = new_title
            record.description = description
            db_session.add(record)
            db_session.flush()
        record_snapshot = {
            "vector_id": record.vector_id,
            "id": record.id,
            "title": record.title,
            "description": record.description,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    metadata_update = {"title": new_title}
    try:
        with _vector_store() as store:
            store.update_book_metadata(book_id, metadata_update)
    except Exception:  # pragma: no cover - defensive logging
        log.warning("books.rename_vector_failed", book_id=book_id, exc_info=True)
    try:
        update_indexed_book_metadata(settings.qdrant_collection, book_id, metadata_update)
    except Exception:  # pragma: no cover - defensive logging
        log.warning("books.rename_manifest_failed", book_id=book_id, exc_info=True)

    entry = index_map.get(book_id)
    if entry is not None:
        current_meta = getattr(entry, "metadata", None)
        if isinstance(current_meta, Mapping):
            merged_meta = dict(current_meta)
            merged_meta["title"] = new_title
            setattr(entry, "metadata", merged_meta)

    updated_record = SimpleNamespace(**record_snapshot)
    return {"book": _serialize_book(updated_record, index_map.get(book_id))}


@router.delete("/{book_id}")
def delete_book(
    book_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> Response:
    owner_external_id = str(current_user.get("id") or "")
    if not owner_external_id:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid user profile")

    with session_scope() as db_session:
        owner = ensure_owner(db_session, current_user)
        repo = repositories.BookRepository(db_session, owner.id)
        record = repo.delete_by_vector_id(book_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Book not found")

    try:
        with _vector_store() as store:
            store.delete_by_book_id(book_id)
    except Exception:  # pragma: no cover - defensive logging
        log.warning("books.delete_vector_failed", book_id=book_id, exc_info=True)
    try:
        remove_indexed_book(settings.qdrant_collection, book_id)
    except Exception:  # pragma: no cover - defensive logging
        log.warning("books.delete_manifest_failed", book_id=book_id, exc_info=True)

    delete_book_files(owner_external_id, book_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = [
    "router",
]
