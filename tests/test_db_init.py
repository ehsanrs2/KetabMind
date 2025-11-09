from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import sessionmaker

from apps.api.db import init as db_init
from apps.api.db import models
import apps.api.db.session as session_module


def test_init_db_creates_tables_and_seeds(tmp_path, monkeypatch):
    db_path = tmp_path / "app.db"
    test_engine = create_engine(f"sqlite:///{db_path}", future=True)
    TestSessionLocal = sessionmaker(bind=test_engine, expire_on_commit=False, autoflush=False)

    monkeypatch.setattr(db_init, "engine", test_engine, raising=False)
    monkeypatch.setattr(db_init, "SessionLocal", TestSessionLocal, raising=False)

    monkeypatch.setattr(session_module, "engine", test_engine, raising=False)
    monkeypatch.setattr(session_module, "_engine", test_engine, raising=False)
    monkeypatch.setattr(session_module, "SessionLocal", TestSessionLocal, raising=False)

    db_init.init_db()
    db_init.init_db()

    inspector = inspect(test_engine)
    tables = inspector.get_table_names()
    assert "users" in tables
    assert "books" in tables

    book_columns = {column["name"] for column in inspector.get_columns("books")}
    assert "vector_id" in book_columns

    with TestSessionLocal() as session:
        alice = session.execute(
            select(models.User).where(models.User.email == "alice@example.com")
        ).scalar_one_or_none()
        assert alice is not None
        bob = session.execute(
            select(models.User).where(models.User.email == "bob@example.com")
        ).scalar_one_or_none()
        assert bob is not None

    assert Path(db_path).is_file()
