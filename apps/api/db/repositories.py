"""Repository helpers scoped to an owner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from . import models


class _BaseRepository:
    """Base repository that enforces owner-level isolation."""

    model: type[Any]

    def __init__(self, session: Session, owner_id: int) -> None:
        self.session = session
        self.owner_id = owner_id

    def _all(self, stmt) -> list[models.Base]:
        return list(self.session.scalars(stmt))

    def _one(self, stmt) -> models.Base | None:
        return self.session.scalar(stmt)


class UserRepository:
    """Simple repository for user management."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_email(self, email: str) -> models.User | None:
        stmt = select(models.User).where(models.User.email == email)
        return self.session.scalar(stmt)

    def create(self, *, email: str, name: str | None = None) -> models.User:
        user = models.User(email=email, name=name)
        self.session.add(user)
        self.session.flush()
        return user


class BookRepository(_BaseRepository):
    model = models.Book

    def list(self) -> list[models.Book]:
        stmt = (
            select(models.Book)
            .where(models.Book.owner_id == self.owner_id)
            .order_by(models.Book.id)
        )
        return self._all(stmt)

    def get(self, book_id: int) -> models.Book | None:
        stmt = select(models.Book).where(
            models.Book.id == book_id, models.Book.owner_id == self.owner_id
        )
        return self._one(stmt)

    def create(self, *, title: str, description: str | None = None) -> models.Book:
        book = models.Book(title=title, description=description, owner_id=self.owner_id)
        self.session.add(book)
        self.session.flush()
        return book


class BookVersionRepository(_BaseRepository):
    model = models.BookVersion

    def list_for_book(self, book_id: int) -> list[models.BookVersion]:
        stmt = (
            select(models.BookVersion)
            .where(
                models.BookVersion.owner_id == self.owner_id,
                models.BookVersion.book_id == book_id,
            )
            .order_by(models.BookVersion.id)
        )
        return self._all(stmt)

    def create(self, *, book_id: int, version: str, notes: str | None = None) -> models.BookVersion:
        book_repo = BookRepository(self.session, self.owner_id)
        book = book_repo.get(book_id)
        if book is None:
            raise ValueError("Book not found or not owned by this user")
        record = models.BookVersion(
            book_id=book.id,
            version=version,
            notes=notes,
            owner_id=self.owner_id,
        )
        self.session.add(record)
        self.session.flush()
        return record


class SessionRepository(_BaseRepository):
    model = models.Session

    def list(
        self,
        *,
        query: str | None = None,
        sort: str = "date_desc",
        include_deleted: bool = False,
    ) -> list[models.Session]:
        stmt = select(models.Session).where(models.Session.owner_id == self.owner_id)
        if not include_deleted:
            stmt = stmt.where(models.Session.deleted_at.is_(None))
        if query:
            search_term = query.strip()
            if search_term:
                pattern = f"%{search_term}%"
                stmt = stmt.where(models.Session.title.ilike(pattern))

        sort_key = sort.lower().strip() if sort else "date_desc"
        if sort_key == "date_asc":
            stmt = stmt.order_by(models.Session.created_at.asc(), models.Session.id.asc())
        elif sort_key == "title_asc":
            stmt = stmt.order_by(func.lower(models.Session.title).asc(), models.Session.id.asc())
        elif sort_key == "title_desc":
            stmt = stmt.order_by(func.lower(models.Session.title).desc(), models.Session.id.asc())
        else:  # default to date_desc
            stmt = stmt.order_by(models.Session.created_at.desc(), models.Session.id.desc())

        return self._all(stmt)

    def get(
        self, session_id: int, *, include_deleted: bool = False
    ) -> models.Session | None:
        stmt = select(models.Session).where(
            models.Session.id == session_id,
            models.Session.owner_id == self.owner_id,
        )
        if not include_deleted:
            stmt = stmt.where(models.Session.deleted_at.is_(None))
        return self._one(stmt)

    def soft_delete(self, session_id: int) -> bool:
        record = self.get(session_id)
        if record is None:
            return False
        if record.deleted_at is not None:
            return False
        record.deleted_at = datetime.now(timezone.utc)
        self.session.add(record)
        return True

    def create(self, *, title: str, book_id: int | None = None) -> models.Session:
        book_ref: models.Book | None = None
        if book_id is not None:
            book_ref = BookRepository(self.session, self.owner_id).get(book_id)
            if book_ref is None:
                raise ValueError("Book not found or not owned by this user")
        session_obj = models.Session(
            title=title,
            book_id=book_ref.id if book_ref is not None else None,
            owner_id=self.owner_id,
        )
        self.session.add(session_obj)
        self.session.flush()
        return session_obj


class BookmarkRepository(_BaseRepository):
    model = models.Bookmark

    def list(self) -> list[models.Bookmark]:
        stmt = (
            select(models.Bookmark)
            .where(models.Bookmark.owner_id == self.owner_id)
            .order_by(models.Bookmark.created_at.desc())
        )
        return self._all(stmt)

    def get(self, bookmark_id: int) -> models.Bookmark | None:
        stmt = select(models.Bookmark).where(
            models.Bookmark.id == bookmark_id,
            models.Bookmark.owner_id == self.owner_id,
        )
        return self._one(stmt)

    def create(self, *, message_id: int) -> models.Bookmark:
        message_stmt = select(models.Message).where(
            models.Message.id == message_id,
            models.Message.owner_id == self.owner_id,
        )
        message = self.session.scalar(message_stmt)
        if message is None:
            raise ValueError("Message not found or not owned by this user")
        if message.role != "assistant":
            raise ValueError("Only assistant messages can be bookmarked")
        existing_stmt = select(models.Bookmark).where(
            models.Bookmark.message_id == message.id,
            models.Bookmark.owner_id == self.owner_id,
        )
        existing = self.session.scalar(existing_stmt)
        if existing is not None:
            return existing
        bookmark = models.Bookmark(
            session_id=message.session_id,
            message_id=message.id,
            owner_id=self.owner_id,
        )
        self.session.add(bookmark)
        self.session.flush()
        return bookmark

    def delete(self, bookmark_id: int) -> bool:
        bookmark = self.get(bookmark_id)
        if bookmark is None:
            return False
        self.session.delete(bookmark)
        return True


class MessageRepository(_BaseRepository):
    model = models.Message

    def list_for_session(self, session_id: int) -> list[models.Message]:
        stmt = (
            select(models.Message)
            .where(
                models.Message.owner_id == self.owner_id,
                models.Message.session_id == session_id,
            )
            .order_by(models.Message.id)
        )
        return self._all(stmt)

    def create(
        self,
        *,
        session_id: int,
        role: str,
        content: str,
        citations: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> models.Message:
        session_repo = SessionRepository(self.session, self.owner_id)
        session_obj = session_repo.get(session_id)
        if session_obj is None:
            raise ValueError("Session not found or not owned by this user")
        message = models.Message(
            session_id=session_obj.id,
            role=role,
            content=content,
            citations=json.dumps(citations) if citations else None,
            meta=json.dumps(meta) if meta else None,
            owner_id=self.owner_id,
        )
        self.session.add(message)
        self.session.flush()
        return message


def delete_sessions_older_than(session: Session, *, older_than: datetime) -> int:
    cutoff_value = older_than
    if isinstance(older_than, datetime) and older_than.tzinfo is not None:
        cutoff_value = older_than.astimezone(timezone.utc)
        bind = session.get_bind()
        if bind is not None and bind.dialect.name == "sqlite":
            cutoff_value = cutoff_value.replace(tzinfo=None)

    stmt = (
        delete(models.Session)
        .where(models.Session.created_at < cutoff_value)
        .execution_options(synchronize_session=False)
    )
    result = session.execute(stmt)
    return int(result.rowcount or 0)


__all__ = [
    "BookmarkRepository",
    "BookRepository",
    "BookVersionRepository",
    "MessageRepository",
    "SessionRepository",
    "UserRepository",
]
