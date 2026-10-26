"""Original analytical PDFs and independently citeable extracted passages."""

from datetime import date, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AnalyticsDocument(Base):
    __tablename__ = "analytics_documents"
    __table_args__ = (
        Index("ix_analytics_documents_queue", "status", "lease_until"),
        Index("ix_analytics_documents_published", "published_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    sha256: Mapped[str] = mapped_column(Text, unique=True)
    filename: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str] = mapped_column(Text)
    published_at: Mapped[date | None] = mapped_column(Date)
    date_origin: Mapped[str] = mapped_column(Text, default="unknown")
    topics: Mapped[list] = mapped_column(JSONB, default=list)
    uploaded_by: Mapped[UUID] = mapped_column(ForeignKey("users.user_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    status: Mapped[str] = mapped_column(Text, default="queued")
    # Immutable original, kept in PostgreSQL for a self-contained first version.
    content: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    parser_version: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[list] = mapped_column(JSONB, default=list)
    error: Mapped[str | None] = mapped_column(Text)
    embedding_error: Mapped[str | None] = mapped_column(Text)
    embedding_model_version: Mapped[str | None] = mapped_column(Text)
    lease_token: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BankAnalyticsSelection(Base):
    __tablename__ = "bank_analytics_selection"
    __table_args__ = (CheckConstraint("id = 1", name="ck_bank_analytics_selection_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_ids: Mapped[list] = mapped_column(JSONB)


class AnalyticsPage(Base):
    __tablename__ = "analytics_pages"

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("analytics_documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    number: Mapped[int] = mapped_column(Integer, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSONB)


class AnalyticsChunk(Base):
    __tablename__ = "analytics_chunks"
    __table_args__ = (Index("ix_analytics_chunks_document", "document_id", "page_number"),)

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("analytics_documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    page_number: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    bbox: Mapped[list] = mapped_column(JSONB)
    search_text: Mapped[str] = mapped_column(Text)
    vector: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    embedding_model_version: Mapped[str | None] = mapped_column(Text)
