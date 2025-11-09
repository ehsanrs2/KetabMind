"""Add vector_id column to books table."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0005_add_book_vector_id"
down_revision = "0004_add_bookmark_tag"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("books", schema=None) as batch_op:
        batch_op.add_column(sa.Column("vector_id", sa.String(length=128), nullable=True))
        batch_op.create_unique_constraint("uq_books_vector_id", ["vector_id"])
        batch_op.create_index("ix_books_vector_id", ["vector_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("books", schema=None) as batch_op:
        batch_op.drop_index("ix_books_vector_id")
        batch_op.drop_constraint("uq_books_vector_id", type_="unique")
        batch_op.drop_column("vector_id")
