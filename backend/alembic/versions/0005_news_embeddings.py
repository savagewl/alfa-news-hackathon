"""News embeddings (pgvector) for relevance ranking and insight generation."""

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0005_news_embeddings"
down_revision = "0004_rss_news"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "news_embeddings",
        sa.Column("article_id", sa.Uuid(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("vector", Vector(1024), nullable=False),
        sa.Column("analysis", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("embedding_model_version", sa.Text(), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("article_id", name="pk_news_embeddings"),
        sa.ForeignKeyConstraint(
            ["article_id"],
            ["news_articles.id"],
            name="fk_news_embeddings_article_id_news_articles",
            ondelete="CASCADE",
        ),
    )
    op.create_index(
        "ix_news_embeddings_vector",
        "news_embeddings",
        ["vector"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"vector": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_news_embeddings_vector", table_name="news_embeddings")
    op.drop_table("news_embeddings")
    # The `vector` extension is left installed: dropping it would break dependent objects.
