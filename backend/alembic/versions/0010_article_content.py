"""Durable article body fetching and explicit provenance.

Revision ID: 0010_article_content
Revises: 0009_news_assessments
"""

import sqlalchemy as sa

from alembic import op

revision = "0010_article_content"
down_revision = "0009_news_assessments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for column in (
        sa.Column("content_origin", sa.Text()),
        sa.Column("content_status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("content_error", sa.Text()),
        sa.Column("content_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("content_next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("content_lease_token", sa.Uuid()),
        sa.Column("content_lease_until", sa.DateTime(timezone=True)),
    ):
        op.add_column("news_articles", column)
    op.create_index("ix_news_articles_content_due", "news_articles",
                    ["content_status", "content_next_attempt_at"])


def downgrade() -> None:
    op.drop_index("ix_news_articles_content_due", table_name="news_articles")
    for name in ("content_lease_until", "content_lease_token", "content_next_attempt_at",
                 "content_attempts", "content_error", "content_status", "content_origin"):
        op.drop_column("news_articles", name)
