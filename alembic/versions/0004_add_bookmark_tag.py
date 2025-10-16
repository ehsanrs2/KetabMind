"""Add tag column to bookmarks."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004_add_bookmark_tag"
down_revision = "0003_add_session_deleted_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "bookmarks",
        sa.Column("tag", sa.String(length=255), nullable=True),
    )
    op.create_index("ix_bookmarks_tag", "bookmarks", ["tag"])


def downgrade() -> None:
    op.drop_index("ix_bookmarks_tag", table_name="bookmarks")
    op.drop_column("bookmarks", "tag")
