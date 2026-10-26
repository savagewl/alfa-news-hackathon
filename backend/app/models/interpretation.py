from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class NewsInterpretation(Base):
    __tablename__ = "news_interpretations"
    __table_args__ = (Index("ix_news_interpretations_company_news", "company_id", "news_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    news_id: Mapped[UUID] = mapped_column(ForeignKey("news_articles.id", ondelete="CASCADE"))
    cache_key: Mapped[str] = mapped_column(Text, unique=True)
    request_fingerprint: Mapped[str | None] = mapped_column(Text, unique=True)
    company_input_fingerprint: Mapped[str | None] = mapped_column(Text)
    news_input_fingerprint: Mapped[str | None] = mapped_column(Text)
    retrieval_fingerprint: Mapped[str | None] = mapped_column(Text)
    model_prompt_version: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    lease_token: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
