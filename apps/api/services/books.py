"""Helper utilities shared across book-related API endpoints."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from apps.api.db import models, repositories
from core.config import settings


def _ensure_upload_root() -> Path:
    upload_dir = settings.upload_dir
    if not upload_dir:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload directory not configured",
        )
    root = Path(upload_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    try:
        return root.resolve()
    except FileNotFoundError:  # pragma: no cover - defensive fallback
        return root


def book_title_from_meta(meta: Mapping[str, Any] | None, fallback: str) -> str:
    if isinstance(meta, Mapping):
        for key in ("title", "subject", "author"):
            value = meta.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
    return fallback


def ensure_book_record(
    db_session: Session,
    owner: models.User,
    *,
    book_id: str,
    metadata: Mapping[str, Any] | None = None,
    title_fallback: str | None = None,
) -> models.Book:
    repo = repositories.BookRepository(db_session, owner.id)
    existing = repo.get_by_vector_id(book_id)
    if existing is not None:
        return existing
    title_source = book_title_from_meta(metadata, title_fallback or book_id)
    record = repo.create(title=title_source, vector_id=book_id)
    db_session.flush()
    return record


def owner_book_directory(owner_external_id: str) -> Path:
    root = _ensure_upload_root()
    return root / owner_external_id


def list_owner_book_ids(owner_external_id: str) -> list[str]:
    root = owner_book_directory(owner_external_id)
    if not root.exists():
        return []
    directories = [p for p in root.iterdir() if p.is_dir()]
    return sorted(path.name for path in directories)


def delete_book_files(owner_external_id: str, book_id: str) -> None:
    book_dir = owner_book_directory(owner_external_id) / book_id
    if book_dir.exists():
        shutil.rmtree(book_dir, ignore_errors=True)


__all__ = [
    "book_title_from_meta",
    "delete_book_files",
    "ensure_book_record",
    "list_owner_book_ids",
    "owner_book_directory",
]
