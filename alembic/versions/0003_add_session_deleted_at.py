"""Add soft delete support to sessions."""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0003_add_session_deleted_at"
down_revision = "0002_add_owner_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_sessions_deleted_at", "sessions", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("ix_sessions_deleted_at", table_name="sessions")
    op.drop_column("sessions", "deleted_at")
