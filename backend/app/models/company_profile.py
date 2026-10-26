"""Persisted company profile inputs and vectors used by background relevance ranking."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CompanyProfileCache(Base):
    __tablename__ = "company_profile_cache"
    __table_args__ = (Index("ix_company_profile_cache_work", "status", "next_attempt_at"),)

    company_id: Mapped[str] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"), primary_key=True
    )
    input_fingerprint: Mapped[str] = mapped_column(Text)
    profile_text: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    fields: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    vectors: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    embedding_model_version: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    lease_token: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
