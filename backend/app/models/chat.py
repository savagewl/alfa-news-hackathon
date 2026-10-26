from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class NewsChatThread(Base):
    __tablename__ = "news_chat_threads"
    __table_args__ = (
        Index(
            "ix_news_chat_threads_owner_context",
            "user_id",
            "company_id",
            "news_id",
            "updated_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"))
    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"))
    news_id: Mapped[UUID] = mapped_column(
        ForeignKey("news_articles.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class NewsChatMessage(Base):
    __tablename__ = "news_chat_messages"
    __table_args__ = (
        UniqueConstraint("thread_id", "client_message_id"),
        UniqueConstraint("reply_to_id"),
        Index("ix_news_chat_messages_thread_order", "thread_id", "created_at", "id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    thread_id: Mapped[UUID] = mapped_column(
        ForeignKey("news_chat_threads.id", ondelete="CASCADE")
    )
    client_message_id: Mapped[UUID | None]
    reply_to_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("news_chat_messages.id", ondelete="CASCADE")
    )
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.user_id", ondelete="SET NULL")
    )
    role: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="ready")
    lease_token: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content: Mapped[str] = mapped_column(Text)
    answer_kind: Mapped[str | None] = mapped_column(Text)
    sources: Mapped[list] = mapped_column(JSONB, default=list)
    warnings: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
