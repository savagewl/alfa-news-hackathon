"""RSS state and a shared, source-attributed news corpus."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class NewsSource(Base):
    __tablename__ = "news_sources"

    code: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    site_url: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text, default="media")


class RssFeed(Base):
    __tablename__ = "rss_feeds"
    __table_args__ = (Index("ix_rss_feeds_due", "enabled", "next_poll_at"),)

    code: Mapped[str] = mapped_column(Text, primary_key=True)
    source_code: Mapped[str] = mapped_column(ForeignKey("news_sources.code"))
    url: Mapped[str] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    poll_interval_seconds: Mapped[int] = mapped_column(Integer)
    next_poll_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    lease_token: Mapped[UUID | None]
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    etag: Mapped[str | None] = mapped_column(Text)
    last_modified: Mapped[str | None] = mapped_column(Text)
    last_body_hash: Mapped[str | None] = mapped_column(Text)
    pending_body_hash: Mapped[str | None] = mapped_column(Text)
    pending_etag: Mapped[str | None] = mapped_column(Text)
    pending_last_modified: Mapped[str | None] = mapped_column(Text)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_new_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)


class RssRun(Base):
    __tablename__ = "rss_runs"
    __table_args__ = (Index("ix_rss_runs_feed_started", "feed_code", "started_at"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    feed_code: Mapped[str] = mapped_column(ForeignKey("rss_feeds.code"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, default="running")
    http_status: Mapped[int | None] = mapped_column(Integer)
    counts: Mapped[dict] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text)


class RssSnapshot(Base):
    __tablename__ = "rss_snapshots"
    __table_args__ = (
        UniqueConstraint("feed_code", "body_hash"),
        Index("ix_rss_snapshots_created", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    feed_code: Mapped[str] = mapped_column(ForeignKey("rss_feeds.code"))
    body_hash: Mapped[str] = mapped_column(Text)
    compressed_body: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class NewsArticle(Base):
    __tablename__ = "news_articles"
    __table_args__ = (
        Index("ix_news_articles_seen", "first_seen_at", "id"),
        Index("ix_news_articles_source_published", "source_code", "published_at"),
        Index("ix_news_articles_topics", "topics", postgresql_using="gin"),
        Index("ix_news_articles_content_due", "content_status", "content_next_attempt_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    source_code: Mapped[str] = mapped_column(ForeignKey("news_sources.code"))
    external_id: Mapped[str | None] = mapped_column(Text)
    original_url: Mapped[str] = mapped_column(Text)
    canonical_url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    content_text: Mapped[str | None] = mapped_column(Text)
    content_origin: Mapped[str | None] = mapped_column(Text)
    content_status: Mapped[str] = mapped_column(Text, default="pending", server_default="pending")
    content_error: Mapped[str | None] = mapped_column(Text)
    content_attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    content_next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_lease_token: Mapped[UUID | None]
    content_lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    authors: Mapped[list] = mapped_column(JSONB)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    source_categories: Mapped[list] = mapped_column(JSONB)
    topics: Mapped[list] = mapped_column(JSONB)
    image_url: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    normalizer_version: Mapped[int] = mapped_column(Integer)


class NewsArticleKey(Base):
    __tablename__ = "news_article_keys"

    source_code: Mapped[str] = mapped_column(ForeignKey("news_sources.code"), primary_key=True)
    kind: Mapped[str] = mapped_column(Text, primary_key=True)
    value_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    article_id: Mapped[UUID] = mapped_column(
        ForeignKey("news_articles.id", ondelete="CASCADE"), index=True
    )


class NewsArticleFeed(Base):
    __tablename__ = "news_article_feeds"

    article_id: Mapped[UUID] = mapped_column(
        ForeignKey("news_articles.id", ondelete="CASCADE"), primary_key=True
    )
    feed_code: Mapped[str] = mapped_column(ForeignKey("rss_feeds.code"), primary_key=True)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class RssItemError(Base):
    __tablename__ = "rss_item_errors"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("rss_runs.id", ondelete="CASCADE"), index=True)
    item_index: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(Text)
    item: Mapped[dict] = mapped_column(JSONB)
