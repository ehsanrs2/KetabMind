from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
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
def owners(db_session: Session) -> tuple[models.User, models.User]:
    user_repo = repositories.UserRepository(db_session)
    owner_one = user_repo.create(email="one@example.com", name="Owner One")
    owner_two = user_repo.create(email="two@example.com", name="Owner Two")
    db_session.commit()
    return owner_one, owner_two


def test_books_are_filtered_by_owner(
    db_session: Session, owners: tuple[models.User, models.User]
) -> None:
    owner_one, owner_two = owners
    repo_one = repositories.BookRepository(db_session, owner_one.id)
    repo_two = repositories.BookRepository(db_session, owner_two.id)

    book_a = repo_one.create(title="Owner 1 Book")
    repo_two.create(title="Owner 2 Book")
    db_session.commit()

    assert [book.title for book in repo_one.list()] == ["Owner 1 Book"]
    assert [book.title for book in repo_two.list()] == ["Owner 2 Book"]

    assert repo_one.get(book_a.id) is not None
    assert repo_one.get(9999) is None


def test_book_versions_respect_ownership(
    db_session: Session, owners: tuple[models.User, models.User]
) -> None:
    owner_one, owner_two = owners
    books_one = repositories.BookRepository(db_session, owner_one.id)
    books_two = repositories.BookRepository(db_session, owner_two.id)
    versions_one = repositories.BookVersionRepository(db_session, owner_one.id)
    versions_two = repositories.BookVersionRepository(db_session, owner_two.id)

    book_one = books_one.create(title="Shared Book")
    book_two = books_two.create(title="Second Book")
    versions_one.create(book_id=book_one.id, version="v1")
    versions_two.create(book_id=book_two.id, version="v1")

    with pytest.raises(ValueError):
        versions_one.create(book_id=book_two.id, version="v2")

    assert [v.version for v in versions_one.list_for_book(book_one.id)] == ["v1"]
    assert versions_one.list_for_book(book_two.id) == []


def test_sessions_are_owner_scoped(
    db_session: Session, owners: tuple[models.User, models.User]
) -> None:
    owner_one, owner_two = owners
    books_one = repositories.BookRepository(db_session, owner_one.id)
    book_one = books_one.create(title="Book")

    sessions_one = repositories.SessionRepository(db_session, owner_one.id)
    sessions_two = repositories.SessionRepository(db_session, owner_two.id)

    session_a = sessions_one.create(title="Session A", book_id=book_one.id)
    sessions_two.create(title="Session B")

    with pytest.raises(ValueError):
        sessions_two.create(title="Cross Session", book_id=book_one.id)

    assert [s.title for s in sessions_one.list()] == ["Session A"]
    assert sessions_one.list() == [session_a]
    assert [s.title for s in sessions_two.list()] == ["Session B"]


def test_bookmarks_filter_by_owner(
    db_session: Session, owners: tuple[models.User, models.User]
) -> None:
    owner_one, owner_two = owners
    bookmark_repo_one = repositories.BookmarkRepository(db_session, owner_one.id)
    bookmark_repo_two = repositories.BookmarkRepository(db_session, owner_two.id)
    sessions_one = repositories.SessionRepository(db_session, owner_one.id)
    sessions_two = repositories.SessionRepository(db_session, owner_two.id)
    session_one = sessions_one.create(title="Session One")
    session_two = sessions_two.create(title="Session Two")

    message_repo_one = repositories.MessageRepository(db_session, owner_one.id)
    message_repo_two = repositories.MessageRepository(db_session, owner_two.id)
    assistant_one = message_repo_one.create(
        session_id=session_one.id, role="assistant", content="Response one"
    )
    assistant_two = message_repo_two.create(
        session_id=session_two.id, role="assistant", content="Response two"
    )

    bookmark_one = bookmark_repo_one.create(message_id=assistant_one.id)
    bookmark_repo_two.create(message_id=assistant_two.id)

    with pytest.raises(ValueError):
        bookmark_repo_one.create(message_id=assistant_two.id)

    user_message = message_repo_one.create(session_id=session_one.id, role="user", content="Hello")
    with pytest.raises(ValueError):
        bookmark_repo_one.create(message_id=user_message.id)

    assert [b.message_id for b in bookmark_repo_one.list()] == [assistant_one.id]
    assert [b.message_id for b in bookmark_repo_two.list()] == [assistant_two.id]
    assert bookmark_repo_one.delete(9999) is False
    assert bookmark_repo_one.delete(bookmark_one.id) is True
    assert bookmark_repo_one.list() == []


def test_bookmarks_filter_by_tag(
    db_session: Session, owners: tuple[models.User, models.User]
) -> None:
    owner_one, _ = owners
    session_repo = repositories.SessionRepository(db_session, owner_one.id)
    session = session_repo.create(title="Session One")
    message_repo = repositories.MessageRepository(db_session, owner_one.id)
    assistant_one = message_repo.create(
        session_id=session.id, role="assistant", content="Response one"
    )
    assistant_two = message_repo.create(
        session_id=session.id, role="assistant", content="Response two"
    )
    assistant_three = message_repo.create(
        session_id=session.id, role="assistant", content="Response three"
    )

    bookmark_repo = repositories.BookmarkRepository(db_session, owner_one.id)
    bookmark_one = bookmark_repo.create(message_id=assistant_one.id, tag="math")
    bookmark_two = bookmark_repo.create(message_id=assistant_two.id, tag="physics")
    bookmark_three = bookmark_repo.create(message_id=assistant_three.id, tag="math")

    all_tags = [b.tag for b in bookmark_repo.list()]
    assert all_tags == ["math", "physics", "math"]

    math_bookmarks = bookmark_repo.list(tag="math")
    assert {b.id for b in math_bookmarks} == {bookmark_one.id, bookmark_three.id}

    physics_bookmarks = bookmark_repo.list(tag="physics")
    assert [b.id for b in physics_bookmarks] == [bookmark_two.id]

    empty_bookmarks = bookmark_repo.list(tag="history")
    assert empty_bookmarks == []


def test_messages_filter_by_owner(
    db_session: Session, owners: tuple[models.User, models.User]
) -> None:
    owner_one, owner_two = owners
    book_one = repositories.BookRepository(db_session, owner_one.id).create(title="Book")
    session_one = repositories.SessionRepository(db_session, owner_one.id).create(
        title="Chat", book_id=book_one.id
    )
    session_two = repositories.SessionRepository(db_session, owner_two.id).create(
        title="Other Chat"
    )

    message_repo_one = repositories.MessageRepository(db_session, owner_one.id)
    message_repo_two = repositories.MessageRepository(db_session, owner_two.id)

    message_repo_one.create(session_id=session_one.id, role="user", content="Hi")
    message_repo_two.create(session_id=session_two.id, role="user", content="Hello")

    with pytest.raises(ValueError):
        message_repo_one.create(session_id=session_two.id, role="user", content="No access")

    assert [m.content for m in message_repo_one.list_for_session(session_one.id)] == ["Hi"]
    assert message_repo_one.list_for_session(session_two.id) == []
