"""Utilities for initializing and seeding the development database."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from sqlalchemy import inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from . import models  # noqa: F401 - ensure models are imported for metadata
from .base import Base
from .session import SessionLocal, engine


def _ensure_directory_exists(db_engine: Engine) -> None:
    url = getattr(db_engine, "url", None)
    if url is None:
        return
    if url.get_backend_name() != "sqlite":  # pragma: no cover - non-sqlite envs
        return
    database = url.database
    if not database:
        return
    path = Path(database)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def ensure_column(
    db_engine: Engine,
    table_name: str,
    column_name: str,
    column_definition_sql: str,
) -> None:
    inspector = inspect(db_engine)
    try:
        columns = {column["name"] for column in inspector.get_columns(table_name)}
    except Exception:  # pragma: no cover - defensive when table missing unexpectedly
        columns = set()
    if column_name in columns:
        return
    statement = text(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition_sql}"
    )
    with db_engine.begin() as connection:
        connection.execute(statement)


def seed_users(session: Session) -> None:
    defaults: Iterable[tuple[str, str | None]] = (
        ("alice@example.com", "Alice"),
        ("bob@example.com", "Bob"),
    )
    for email, name in defaults:
        result = session.execute(
            select(models.User).where(models.User.email == email.lower())
        )
        user = result.scalar_one_or_none()
        if user is None:
            session.add(models.User(email=email.lower(), name=name))


def init_db() -> None:
    _ensure_directory_exists(engine)
    Base.metadata.create_all(bind=engine)
    ensure_column(engine, "books", "vector_id", "VARCHAR(128)")
    session = SessionLocal()
    try:
        seed_users(session)
        session.commit()
    except Exception:  # pragma: no cover - defensive rollback on failure
        session.rollback()
        raise
    finally:
        session.close()


__all__ = ["ensure_column", "init_db", "seed_users"]
