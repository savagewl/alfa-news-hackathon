import base64
import binascii
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import AwareDatetime
from sqlalchemy import select, tuple_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.db.session import SessionDep
from app.models.news import NewsArticle, NewsSource, RssFeed
from app.schemas.news import (
    ArticleResponse,
    FeedResponse,
    NewsError,
    NewsPage,
    NewsTopic,
    SourceResponse,
)

router = APIRouter(
    prefix="/news",
    tags=["News"],
    dependencies=[Depends(get_current_user)],
    responses={
        401: {"model": NewsError, "description": "Требуется действующий JWT"},
        503: {"model": NewsError, "description": "База данных недоступна"},
    },
)


async def news_session(session: SessionDep) -> AsyncIterator[AsyncSession]:
    try:
        yield session
    except (SQLAlchemyError, OSError, TimeoutError):
        await session.rollback()
        raise HTTPException(503, "Database unavailable") from None


NewsSession = Annotated[AsyncSession, Depends(news_session)]


def encode_cursor(article: NewsArticle) -> str:
    content = json.dumps([article.first_seen_at.isoformat(), str(article.id)]).encode()
    return base64.urlsafe_b64encode(content).decode().rstrip("=")


def decode_cursor(value: str) -> tuple[datetime, UUID]:
    try:
        decoded = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        payload = json.loads(decoded)
        if (
            not isinstance(payload, list)
            or len(payload) != 2
            or not all(isinstance(part, str) for part in payload)
        ):
            raise ValueError("Invalid cursor payload")
        timestamp, identifier = payload
        moment = datetime.fromisoformat(timestamp)
        if moment.tzinfo is None:
            raise ValueError("Cursor needs timezone")
        return moment, UUID(identifier)
    except (ValueError, TypeError, UnicodeError, binascii.Error, OverflowError):
        raise HTTPException(422, "Invalid news cursor") from None


@router.get("", response_model=NewsPage, summary="Получить новости из всех источников")
async def list_news(
    session: NewsSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=512)] = None,
    source: Annotated[str | None, Query(max_length=100)] = None,
    topic: NewsTopic | None = None,
    published_from: AwareDatetime | None = None,
    published_to: AwareDatetime | None = None,
    seen_from: AwareDatetime | None = None,
    seen_to: AwareDatetime | None = None,
) -> NewsPage:
    """Общий корпус. Сортировка по обнаружению (новые первыми), курсор — first_seen_at + id.

    Даты фильтров включительные, обязательно с часовым поясом. Описание и дата публикации
    могут быть null. Публикации разных изданий об одном событии сохраняются отдельно.
    """
    query = select(NewsArticle)
    if source:
        query = query.where(NewsArticle.source_code == source)
    if topic:
        query = query.where(NewsArticle.topics.contains([topic]))
    for field, lower, upper in (
        (NewsArticle.published_at, published_from, published_to),
        (NewsArticle.first_seen_at, seen_from, seen_to),
    ):
        if lower and upper and lower > upper:
            raise HTTPException(422, "Period start must precede period end")
        if lower:
            query = query.where(field >= lower)
        if upper:
            query = query.where(field <= upper)
    if cursor:
        query = query.where(
            tuple_(NewsArticle.first_seen_at, NewsArticle.id) < tuple_(*decode_cursor(cursor))
        )
    query = query.order_by(NewsArticle.first_seen_at.desc(), NewsArticle.id.desc()).limit(limit + 1)
    records = list((await session.scalars(query)).all())
    more = len(records) > limit
    records = records[:limit]
    return NewsPage(
        items=[ArticleResponse.model_validate(record) for record in records],
        next_cursor=encode_cursor(records[-1]) if more else None,
    )


@router.get("/sources", response_model=list[SourceResponse], summary="Источники и состояние RSS")
async def list_sources(session: NewsSession) -> list[SourceResponse]:
    sources = (await session.scalars(select(NewsSource).order_by(NewsSource.code))).all()
    feeds = (await session.scalars(select(RssFeed).order_by(RssFeed.code))).all()
    now = datetime.now(UTC)
    result = []
    for source in sources:
        statuses = []
        for feed in feeds:
            if feed.source_code != source.code:
                continue
            stale = feed.last_new_at is not None and now - feed.last_new_at > timedelta(days=1)
            overdue = now > feed.next_poll_at + timedelta(seconds=feed.poll_interval_seconds * 3)
            values = {
                field: getattr(feed, field)
                for field in FeedResponse.model_fields
                if field != "needs_attention"
            }
            statuses.append(
                FeedResponse(
                    **values,
                    needs_attention=feed.enabled
                    and (stale or overdue or feed.consecutive_failures >= 3),
                )
            )
        result.append(
            SourceResponse(
                code=source.code,
                name=source.name,
                site_url=source.site_url,
                kind=source.kind,
                feeds=statuses,
            )
        )
    return result


@router.get(
    "/{news_id}",
    response_model=ArticleResponse,
    summary="Получить публикацию",
    responses={404: {"model": NewsError, "description": "Публикация не найдена"}},
)
async def get_news(news_id: UUID, session: NewsSession) -> ArticleResponse:
    record = await session.get(NewsArticle, news_id)
    if record is None:
        raise HTTPException(404, "News article not found")
    return ArticleResponse.model_validate(record)
