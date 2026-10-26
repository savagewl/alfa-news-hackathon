"""Leased article fetch jobs; no database transaction spans an HTTP request."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models.news import NewsArticle
from app.services.rss.parser import ParsedArticle


@dataclass(frozen=True)
class ArticleContentClaim:
    article_id: UUID
    source_code: str
    url: str
    token: UUID


def set_article_content(article: NewsArticle, text: str, origin: str) -> None:
    article.content_text = text
    # The shared hash invalidates news embeddings, assessments and interpretations.
    parsed = ParsedArticle(**{
        name: getattr(article, name) for name in ParsedArticle.__dataclass_fields__
    })
    article.content_hash = parsed.values()["content_hash"]
    article.content_origin = origin
    article.content_status = "ready"
    article.content_error = None
    article.content_next_attempt_at = None
    article.content_lease_token = None
    article.content_lease_until = None
    article.updated_at = datetime.now(UTC)


async def claim_article_content(
    factory: async_sessionmaker, source_codes: list[str]
) -> ArticleContentClaim | None:
    async with factory.begin() as session:
        article = await session.scalar(
            select(NewsArticle)
            .where(
                NewsArticle.source_code.in_(source_codes),
                or_(
                    NewsArticle.content_status == "pending",
                    and_(NewsArticle.content_status == "retry",
                         NewsArticle.content_next_attempt_at <= func.now()),
                    and_(NewsArticle.content_status == "processing",
                         NewsArticle.content_lease_until < func.now()),
                ),
            )
            .order_by(NewsArticle.first_seen_at.desc(), NewsArticle.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        if article is None:
            return None
        article.content_status = "processing"
        article.content_attempts += 1
        article.content_lease_token = uuid4()
        article.content_lease_until = datetime.now(UTC) + timedelta(seconds=120)
        return ArticleContentClaim(article.id, article.source_code,
                                   article.canonical_url, article.content_lease_token)


async def finish_article_content(
    factory: async_sessionmaker, claim: ArticleContentClaim, *,
    text: str | None = None, error: str | None = None,
) -> bool:
    async with factory.begin() as session:
        article = await session.scalar(
            select(NewsArticle)
            .where(NewsArticle.id == claim.article_id,
                   NewsArticle.content_lease_token == claim.token,
                   NewsArticle.content_lease_until > func.now())
            .with_for_update()
        )
        if article is None:
            return False
        if text:
            set_article_content(article, text, "article")
        else:
            article.content_error = (error or "Article body unavailable")[:500]
            article.content_status = "unavailable" if article.content_attempts >= 5 else "retry"
            article.content_next_attempt_at = (
                None if article.content_status == "unavailable" else
                datetime.now(UTC) + timedelta(seconds=min(60 * 2**article.content_attempts, 3600))
            )
            article.content_lease_token = None
            article.content_lease_until = None
        return True
