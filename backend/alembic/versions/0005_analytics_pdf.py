"""Persistent analytical PDFs, extraction queue and citeable passages.

Revision ID: 0005_analytics_pdf
Revises: 0004_rss_news
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as pg

from alembic import op

revision = "0005_analytics_pdf"
down_revision = "0004_rss_news"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "analytics_documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("sha256", sa.Text(), nullable=False, unique=True),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("publisher", sa.Text(), nullable=False),
        sa.Column("published_at", sa.Date()),
        sa.Column("date_origin", sa.Text(), nullable=False),
        sa.Column("topics", pg.JSONB(), nullable=False),
        sa.Column("uploaded_by", sa.Uuid(), sa.ForeignKey("users.user_id"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("parser_version", sa.Text()),
        sa.Column("warnings", pg.JSONB(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("embedding_error", sa.Text()),
        sa.Column("embedding_model_version", sa.Text()),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_analytics_documents_queue",
        "analytics_documents",
        ["status", "lease_until"],
    )
    op.create_index("ix_analytics_documents_published", "analytics_documents", ["published_at"])
    op.create_table(
        "analytics_pages",
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("analytics_documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("number", sa.Integer(), primary_key=True),
        sa.Column("payload", pg.JSONB(), nullable=False),
    )
    op.create_table(
        "analytics_chunks",
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("analytics_documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("chunk_id", sa.Text(), primary_key=True),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("section", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("bbox", pg.JSONB(), nullable=False),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("vector", pg.ARRAY(sa.Float())),
        sa.Column("embedding_model_version", sa.Text()),
    )
    op.create_index(
        "ix_analytics_chunks_document",
        "analytics_chunks",
        ["document_id", "page_number"],
    )
    op.execute(
        "CREATE INDEX ix_analytics_chunks_fts ON analytics_chunks "
        "USING gin (to_tsvector('russian', search_text))"
    )


def downgrade():
    op.drop_table("analytics_chunks")
    op.drop_table("analytics_pages")
    op.drop_table("analytics_documents")
