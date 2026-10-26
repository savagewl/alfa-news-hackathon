"""Persist the shared bank analytics selection independently of environment settings."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0011_bank_analytics_selection"
down_revision = "0010_article_content"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "bank_analytics_selection",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_ids", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_bank_analytics_selection_singleton"),
    )


def downgrade() -> None:
    op.drop_table("bank_analytics_selection")
