"""Add owner scoping columns and backfill with a dev user."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002_add_owner_id"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None

_TEST_USER_EMAIL = "dev@example.com"
_TEST_USER_NAME = "Development User"


def upgrade() -> None:
    op.add_column("books", sa.Column("owner_id", sa.Integer(), nullable=True))
    op.create_index("ix_books_owner_id", "books", ["owner_id"])
    op.create_foreign_key(
        "fk_books_owner_id_users", "books", "users", ["owner_id"], ["id"], ondelete="CASCADE"
    )

    op.add_column("book_versions", sa.Column("owner_id", sa.Integer(), nullable=True))
    op.create_index("ix_book_versions_owner_id", "book_versions", ["owner_id"])
    op.create_foreign_key(
        "fk_book_versions_owner_id_users",
        "book_versions",
        "users",
        ["owner_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.add_column("sessions", sa.Column("owner_id", sa.Integer(), nullable=True))
    op.create_index("ix_sessions_owner_id", "sessions", ["owner_id"])
    op.create_foreign_key(
        "fk_sessions_owner_id_users",
        "sessions",
        "users",
        ["owner_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.add_column("bookmarks", sa.Column("owner_id", sa.Integer(), nullable=True))
    op.create_index("ix_bookmarks_owner_id", "bookmarks", ["owner_id"])
    op.create_foreign_key(
        "fk_bookmarks_owner_id_users",
        "bookmarks",
        "users",
        ["owner_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.add_column("messages", sa.Column("owner_id", sa.Integer(), nullable=True))
    op.create_index("ix_messages_owner_id", "messages", ["owner_id"])
    op.create_foreign_key(
        "fk_messages_owner_id_users",
        "messages",
        "users",
        ["owner_id"],
        ["id"],
        ondelete="CASCADE",
    )

    bind = op.get_bind()
    bind.execute(
        sa.text(
            "INSERT INTO users (email, name, created_at, updated_at)"
            " SELECT :email, :name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP"
            " WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = :email)"
        ),
        {"email": _TEST_USER_EMAIL, "name": _TEST_USER_NAME},
    )
    user_id = bind.execute(
        sa.text("SELECT id FROM users WHERE email = :email"),
        {"email": _TEST_USER_EMAIL},
    ).scalar_one()

    for table in ("books", "book_versions", "sessions", "bookmarks", "messages"):
        bind.execute(
            sa.text(f"UPDATE {table} SET owner_id = :user_id WHERE owner_id IS NULL"),
            {"user_id": user_id},
        )

    op.alter_column("books", "owner_id", nullable=False)
    op.alter_column("book_versions", "owner_id", nullable=False)
    op.alter_column("sessions", "owner_id", nullable=False)
    op.alter_column("bookmarks", "owner_id", nullable=False)
    op.alter_column("messages", "owner_id", nullable=False)


def downgrade() -> None:
    op.drop_constraint("fk_messages_owner_id_users", "messages", type_="foreignkey")
    op.drop_index("ix_messages_owner_id", table_name="messages")
    op.drop_column("messages", "owner_id")

    op.drop_constraint("fk_bookmarks_owner_id_users", "bookmarks", type_="foreignkey")
    op.drop_index("ix_bookmarks_owner_id", table_name="bookmarks")
    op.drop_column("bookmarks", "owner_id")

    op.drop_constraint("fk_sessions_owner_id_users", "sessions", type_="foreignkey")
    op.drop_index("ix_sessions_owner_id", table_name="sessions")
    op.drop_column("sessions", "owner_id")

    op.drop_constraint("fk_book_versions_owner_id_users", "book_versions", type_="foreignkey")
    op.drop_index("ix_book_versions_owner_id", table_name="book_versions")
    op.drop_column("book_versions", "owner_id")

    op.drop_constraint("fk_books_owner_id_users", "books", type_="foreignkey")
    op.drop_index("ix_books_owner_id", table_name="books")
    op.drop_column("books", "owner_id")
