"""Repository helpers scoped to an owner."""

from __future__ import annotations

from typing import Any, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models


class _BaseRepository:
    """Base repository that enforces owner-level isolation."""

    model: type[Any]

    def __init__(self, session: Session, owner_id: int) -> None:
        self.session = session
        self.owner_id = owner_id

    def _all(self, stmt) -> List[models.Base]:
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

    def list(self) -> List[models.Book]:
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

    def list_for_book(self, book_id: int) -> List[models.BookVersion]:
        stmt = (
            select(models.BookVersion)
            .where(
                models.BookVersion.owner_id == self.owner_id,
                models.BookVersion.book_id == book_id,
            )
            .order_by(models.BookVersion.id)
        )
        return self._all(stmt)

    def create(
        self, *, book_id: int, version: str, notes: str | None = None
    ) -> models.BookVersion:
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

    def list(self) -> List[models.Session]:
        stmt = (
            select(models.Session)
            .where(models.Session.owner_id == self.owner_id)
            .order_by(models.Session.id)
        )
        return self._all(stmt)

    def get(self, session_id: int) -> models.Session | None:
        stmt = select(models.Session).where(
            models.Session.id == session_id,
            models.Session.owner_id == self.owner_id,
        )
        return self._one(stmt)

    def create(
        self, *, title: str, book_id: int | None = None
    ) -> models.Session:
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

    def list_for_book(self, book_id: int) -> List[models.Bookmark]:
        stmt = (
            select(models.Bookmark)
            .where(
                models.Bookmark.owner_id == self.owner_id,
                models.Bookmark.book_id == book_id,
            )
            .order_by(models.Bookmark.page)
        )
        return self._all(stmt)

    def create(
        self, *, book_id: int, page: int, note: str | None = None
    ) -> models.Bookmark:
        book = BookRepository(self.session, self.owner_id).get(book_id)
        if book is None:
            raise ValueError("Book not found or not owned by this user")
        bookmark = models.Bookmark(
            book_id=book.id,
            page=page,
            note=note,
            owner_id=self.owner_id,
        )
        self.session.add(bookmark)
        self.session.flush()
        return bookmark


class MessageRepository(_BaseRepository):
    model = models.Message

    def list_for_session(self, session_id: int) -> List[models.Message]:
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
        self, *, session_id: int, role: str, content: str
    ) -> models.Message:
        session_repo = SessionRepository(self.session, self.owner_id)
        session_obj = session_repo.get(session_id)
        if session_obj is None:
            raise ValueError("Session not found or not owned by this user")
        message = models.Message(
            session_id=session_obj.id,
            role=role,
            content=content,
            owner_id=self.owner_id,
        )
        self.session.add(message)
        self.session.flush()
        return message


__all__ = [
    "BookmarkRepository",
    "BookRepository",
    "BookVersionRepository",
    "MessageRepository",
    "SessionRepository",
    "UserRepository",
]
