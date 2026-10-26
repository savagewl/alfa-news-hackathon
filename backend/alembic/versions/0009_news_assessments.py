"""Short company news assessments queue and cache.

Revision ID: 0009_news_assessments
Revises: 0008_ai_background_state
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0009_news_assessments"
down_revision = "0008_ai_background_state"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "news_assessments",
        sa.Column(
            "company_id",
            sa.Text(),
            sa.ForeignKey("companies.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "news_id",
            sa.Uuid(),
            sa.ForeignKey("news_articles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("input_fingerprint", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("last_error", sa.Text()),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("result", JSONB()),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_news_assessments_work", "news_assessments", ["status", "next_attempt_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_news_assessments_work", table_name="news_assessments")
    op.drop_table("news_assessments")
