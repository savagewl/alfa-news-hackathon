"""Cache on-demand company/news interpretations independently of source articles."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0006_news_interpretations"
down_revision = "0005_analytics_pdf"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "news_interpretations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "company_id",
            sa.Text(),
            sa.ForeignKey("companies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "news_id",
            sa.Uuid(),
            sa.ForeignKey("news_articles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cache_key", sa.Text(), unique=True, nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("result", JSONB()),
        sa.Column("error", sa.Text()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_news_interpretations_company_news", "news_interpretations", ["company_id", "news_id"]
    )


def downgrade():
    op.drop_table("news_interpretations")
