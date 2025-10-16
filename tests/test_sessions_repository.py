from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from apps.api.db import models, repositories
from apps.api.db.base import Base


@pytest.fixture(scope="module")
def engine() -> Iterator:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(engine) -> Iterator[Session]:
    connection = engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection, expire_on_commit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def owner(db_session: Session) -> models.User:
    user_repo = repositories.UserRepository(db_session)
    owner = user_repo.create(email="owner@example.com", name="Owner")
    db_session.commit()
    return owner


def test_session_search_soft_delete_and_cleanup(db_session: Session, owner: models.User) -> None:
    repo = repositories.SessionRepository(db_session, owner.id)

    session_alpha = repo.create(title="Alpha Exploration")
    session_beta = repo.create(title="Beta Testing")
    session_gamma = repo.create(title="Gamma Review")
    session_old = repo.create(title="Old Memories")
    db_session.commit()

    old_timestamp = datetime.now(UTC) - timedelta(days=45)
    session_old.created_at = old_timestamp
    session_old.updated_at = old_timestamp
    db_session.add(session_old)
    db_session.commit()

    search_results = repo.list(query="beta")
    assert [item.title for item in search_results] == ["Beta Testing"]

    assert repo.soft_delete(session_beta.id) is True
    db_session.commit()
    assert repo.soft_delete(session_beta.id) is False

    remaining_titles = [item.title for item in repo.list(sort="title_asc")]
    assert remaining_titles == ["Alpha Exploration", "Gamma Review", "Old Memories"]

    total_rows = db_session.scalar(select(func.count()).select_from(models.Session))
    assert total_rows == 4

    beta_record = repo.get(session_beta.id, include_deleted=True)
    assert beta_record is not None and beta_record.deleted_at is not None
    assert repo.list(query="beta") == []

    cutoff = datetime.now(UTC) - timedelta(days=30)
    removed = repositories.delete_sessions_older_than(db_session, older_than=cutoff)
    db_session.commit()

    assert removed == 1

    active_titles = [item.title for item in repo.list(sort="date_desc")]
    assert set(active_titles) == {"Alpha Exploration", "Gamma Review"}

    remaining_total = db_session.scalar(select(func.count()).select_from(models.Session))
    assert remaining_total == 3

    assert repo.get(session_alpha.id) is not None
    assert repo.get(session_gamma.id) is not None
