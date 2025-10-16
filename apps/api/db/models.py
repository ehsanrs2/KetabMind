"""SQLAlchemy ORM models used by the API."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class TimestampMixin:
    """Mixin that tracks creation and update timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )


class User(TimestampMixin, Base):
    """Application user model."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    books: Mapped[list[Book]] = relationship(back_populates="owner")
    sessions: Mapped[list[Session]] = relationship(back_populates="owner")
    bookmarks: Mapped[list[Bookmark]] = relationship(back_populates="owner")
    messages: Mapped[list[Message]] = relationship(back_populates="owner")


class Book(TimestampMixin, Base):
    """Indexed book."""

    __tablename__ = "books"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    owner: Mapped[User] = relationship(back_populates="books")
    versions: Mapped[list[BookVersion]] = relationship(
        back_populates="book", cascade="all, delete-orphan"
    )
    sessions: Mapped[list[Session]] = relationship(
        back_populates="book", cascade="all, delete-orphan"
    )


class BookVersion(TimestampMixin, Base):
    """Specific version of a book."""

    __tablename__ = "book_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    book_id: Mapped[int] = mapped_column(
        ForeignKey("books.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    book: Mapped[Book] = relationship(back_populates="versions")
    owner: Mapped[User] = relationship()


class Session(TimestampMixin, Base):
    """Chat session for a user."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    book_id: Mapped[int | None] = mapped_column(
        ForeignKey("books.id", ondelete="SET NULL"), nullable=True, index=True
    )
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    owner: Mapped[User] = relationship(back_populates="sessions")
    book: Mapped[Book | None] = relationship(back_populates="sessions")
    messages: Mapped[list[Message]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    bookmarks: Mapped[list[Bookmark]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Bookmark(TimestampMixin, Base):
    """Saved location inside a book."""

    __tablename__ = "bookmarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tag: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    owner: Mapped[User] = relationship(back_populates="bookmarks")
    session: Mapped[Session] = relationship(back_populates="bookmarks")
    message: Mapped[Message] = relationship(back_populates="bookmark")


class Message(TimestampMixin, Base):
    """Conversation message inside a session."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    owner: Mapped[User] = relationship(back_populates="messages")
    session: Mapped[Session] = relationship(back_populates="messages")
    bookmark: Mapped[Bookmark | None] = relationship(
        back_populates="message", cascade="all, delete-orphan", uselist=False
    )


__all__ = [
    "Bookmark",
    "Book",
    "BookVersion",
    "Message",
    "Session",
    "TimestampMixin",
    "User",
]
