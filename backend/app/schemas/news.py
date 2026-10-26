from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

NewsTopic = Literal[
    "economy", "politics", "business", "finance", "taxes", "regulation", "technology", "other"
]


class NewsError(BaseModel):
    detail: str


class ArticleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_code: str
    external_id: str | None
    original_url: str
    canonical_url: str
    title: str
    summary: str | None
    content_text: str | None
    content_origin: Literal["rss", "article"] | None = None
    content_status: Literal["pending", "processing", "retry", "ready", "unavailable"] = "pending"
    content_error: str | None = None
    authors: list[str]
    published_at: datetime | None
    source_updated_at: datetime | None
    first_seen_at: datetime
    last_seen_at: datetime
    updated_at: datetime
    source_categories: list[str]
    topics: list[str]
    image_url: str | None
    normalizer_version: int


class NewsPage(BaseModel):
    items: list[ArticleResponse]
    next_cursor: str | None


class FeedResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    code: str
    url: str
    enabled: bool
    poll_interval_seconds: int
    next_poll_at: datetime
    last_checked_at: datetime | None
    last_success_at: datetime | None
    last_new_at: datetime | None
    consecutive_failures: int
    last_error: str | None
    needs_attention: bool


class SourceResponse(BaseModel):
    code: str
    name: str
    site_url: str
    kind: str
    feeds: list[FeedResponse]
