"""Vector representation and LLM summary of news for the AI relevance services."""

from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# Размерность BGE-M3; должна совпадать с AI_EMBED_DIM и миграцией 0005.
EMBEDDING_DIM = 1024


class NewsEmbedding(Base):
    __tablename__ = "news_embeddings"
    __table_args__ = (
        Index(
            "ix_news_embeddings_vector",
            "vector",
            postgresql_using="hnsw",
            postgresql_ops={"vector": "vector_cosine_ops"},
        ),
        Index(
            "ix_news_embeddings_work", "status", "next_attempt_at", "lease_until"
        ),
    )

    article_id: Mapped[UUID] = mapped_column(
        ForeignKey("news_articles.id", ondelete="CASCADE"), primary_key=True
    )
    # Pending/retry rows double as the durable in-process worker queue.  The
    # payload is populated atomically only after both LLM and BGE succeed.
    summary: Mapped[str | None] = mapped_column(Text)
    vector: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    analysis: Mapped[dict] = mapped_column(JSONB)
    source_content_hash: Mapped[str] = mapped_column(Text, default="")
    input_fingerprint: Mapped[str] = mapped_column(Text, default="")
    llm_model_version: Mapped[str] = mapped_column(Text, default="")
    prompt_version: Mapped[str] = mapped_column(Text, default="")
    embedding_model_version: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(Text, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    lease_token: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
