import hashlib
import random
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import delete, func, or_, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.models.news import (
    NewsArticle,
    NewsArticleFeed,
    NewsArticleKey,
    NewsSource,
    RssFeed,
    RssItemError,
    RssRun,
    RssSnapshot,
)
from app.services.rss.catalog import FeedConfig
from app.services.rss.fetcher import FetchResult
from app.services.rss.parser import ParsedArticle, ParsedFeed


@dataclass(frozen=True)
class FeedClaim:
    feed_code: str
    token: UUID
    run_id: UUID
    etag: str | None
    last_modified: str | None
    pending_body: bytes | None = None
    pending_etag: str | None = None
    pending_last_modified: str | None = None


class LeaseLostError(Exception):
    pass


async def sync_catalog(factory: async_sessionmaker, configs: tuple[FeedConfig, ...]) -> None:
    async with factory.begin() as session:
        # Stable ordering prevents concurrent application startups deadlocking on catalog rows.
        for config in sorted(configs, key=lambda c: (c.code, c.feed_code)):
            statement = insert(NewsSource).values(
                code=config.code, name=config.name, site_url=config.site_url, kind=config.kind
            )
            await session.execute(
                statement.on_conflict_do_update(
                    index_elements=[NewsSource.code],
                    set_={k: getattr(statement.excluded, k) for k in ("name", "site_url", "kind")},
                )
            )
            statement = insert(RssFeed).values(
                code=config.feed_code,
                source_code=config.code,
                url=config.url,
                enabled=config.enabled,
                poll_interval_seconds=config.interval,
            )
            # A changed endpoint must be fetched without the old endpoint's validators/snapshot.
            await session.execute(
                update(RssFeed)
                .where(RssFeed.code == config.feed_code, RssFeed.url != config.url)
                .values(
                    etag=None,
                    last_modified=None,
                    last_body_hash=None,
                    pending_body_hash=None,
                    pending_etag=None,
                    pending_last_modified=None,
                    next_poll_at=func.now(),
                    lease_token=None,
                    lease_until=None,
                )
            )
            await session.execute(
                statement.on_conflict_do_update(
                    index_elements=[RssFeed.code],
                    set_={
                        k: getattr(statement.excluded, k)
                        for k in ("url", "enabled", "poll_interval_seconds")
                    },
                )
            )


async def claim_feed(
    factory: async_sessionmaker, configs: tuple[FeedConfig, ...], lease_seconds: int
) -> FeedClaim | None:
    async with factory.begin() as session:
        feed = await session.scalar(
            select(RssFeed)
            .where(
                RssFeed.enabled.is_(True),
                RssFeed.code.in_([c.feed_code for c in configs if c.enabled]),
                RssFeed.next_poll_at <= func.now(),
                or_(RssFeed.lease_until.is_(None), RssFeed.lease_until < func.now()),
            )
            .order_by(RssFeed.next_poll_at, RssFeed.code)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if feed is None:
            return None
        now = datetime.now(UTC)
        feed.lease_token = uuid4()
        feed.lease_until = now + timedelta(seconds=lease_seconds)
        await session.execute(
            update(RssRun)
            .where(RssRun.feed_code == feed.code, RssRun.status == "running")
            .values(status="abandoned", finished_at=now, error="Worker lease expired")
        )
        run = RssRun(id=uuid4(), feed_code=feed.code)
        session.add(run)
        body = None
        if feed.pending_body_hash:
            body = await session.scalar(
                select(RssSnapshot.compressed_body).where(
                    RssSnapshot.feed_code == feed.code,
                    RssSnapshot.body_hash == feed.pending_body_hash,
                )
            )
        return FeedClaim(
            feed.code,
            feed.lease_token,
            run.id,
            feed.etag,
            feed.last_modified,
            body,
            feed.pending_etag,
            feed.pending_last_modified,
        )


async def owned_feed(session: AsyncSession, claim: FeedClaim) -> RssFeed:
    feed = await session.scalar(
        select(RssFeed)
        .where(
            RssFeed.code == claim.feed_code,
            RssFeed.lease_token == claim.token,
            RssFeed.lease_until > func.now(),
        )
        .with_for_update()
    )
    if feed is None:
        raise LeaseLostError("Feed lease is no longer owned")
    return feed


async def renew_lease(factory: async_sessionmaker, claim: FeedClaim, seconds: int) -> None:
    async with factory.begin() as session:
        feed = await owned_feed(session, claim)
        feed.lease_until = datetime.now(UTC) + timedelta(seconds=seconds)


async def save_snapshot(factory: async_sessionmaker, claim: FeedClaim, result: FetchResult) -> None:
    body_hash = hashlib.sha256(result.body).hexdigest()
    compressed = zlib.compress(result.body)
    async with factory.begin() as session:
        feed = await owned_feed(session, claim)
        await session.execute(
            insert(RssSnapshot)
            .values(feed_code=claim.feed_code, body_hash=body_hash, compressed_body=compressed)
            .on_conflict_do_nothing(index_elements=["feed_code", "body_hash"])
        )
        feed.pending_body_hash = body_hash
        feed.pending_etag = result.etag
        feed.pending_last_modified = result.last_modified


def article_keys(article: ParsedArticle, config: FeedConfig) -> list[tuple[str, str]]:
    keys = [("url", article.canonical_url)]
    if article.external_id:
        scope = config.code if config.guid_scope == "source" else config.feed_code
        keys.append((f"id:{scope}", article.external_id))
    return [(kind, hashlib.sha256(value.encode()).hexdigest()) for kind, value in keys]


async def save_article(
    session: AsyncSession, article: ParsedArticle, config: FeedConfig, now: datetime
) -> str:
    keys = article_keys(article, config)
    ids = set(
        (
            await session.scalars(
                select(NewsArticleKey.article_id).where(
                    NewsArticleKey.source_code == config.code,
                    tuple_(NewsArticleKey.kind, NewsArticleKey.value_hash).in_(keys),
                )
            )
        ).all()
    )
    if len(ids) > 1:
        # Never silently merge two pre-existing publications with conflicting identities.
        raise ValueError("Article identity refers to different existing publications")
    record = (
        await session.scalar(
            select(NewsArticle).where(NewsArticle.id == next(iter(ids))).with_for_update()
        ) if ids else None
    )
    values = article.values()
    if record is None:
        record = NewsArticle(
            id=uuid4(),
            source_code=config.code,
            first_seen_at=now,
            last_seen_at=now,
            updated_at=now,
            content_origin="rss" if article.content_text else None,
            content_status="ready" if article.content_text else "pending",
            **values,
        )
        session.add(record)
        await session.flush()
        outcome = "new"
    else:
        outcome = "unchanged"
        body_needs_refresh = (
            record.content_origin == "article"
            and article.source_updated_at is not None
            and (record.source_updated_at is None
                 or article.source_updated_at > record.source_updated_at)
        )
        # Sparse RSS must not erase a downloaded body, a previous summary or date.
        for key in ("summary", "content_text", "published_at", "source_updated_at", "image_url"):
            if values[key] is None:
                values[key] = getattr(record, key)
        for key in ("authors", "source_categories"):
            if not values[key]:
                values[key] = getattr(record, key)
        if values["topics"] == ["other"] and record.topics != ["other"]:
            values["topics"] = record.topics
        merged = ParsedArticle(**{k: v for k, v in values.items() if k != "content_hash"})
        values = merged.values()
        if values["content_hash"] != record.content_hash:
            for key, value in values.items():
                setattr(record, key, value)
            record.updated_at = now
            outcome = "updated"
        if article.content_text:
            record.content_origin = "rss"
            record.content_status = "ready"
            record.content_error = None
            record.content_lease_token = None
            record.content_lease_until = None
            record.content_next_attempt_at = None
        elif body_needs_refresh:
            record.content_status = "pending"
            record.content_attempts = 0
            record.content_error = None
            record.content_lease_token = None
            record.content_lease_until = None
            record.content_next_attempt_at = None
        record.last_seen_at = now
    for kind, value_hash in keys:
        await session.execute(
            insert(NewsArticleKey)
            .values(source_code=config.code, kind=kind, value_hash=value_hash, article_id=record.id)
            .on_conflict_do_nothing()
        )
    statement = insert(NewsArticleFeed).values(
        article_id=record.id, feed_code=config.feed_code, last_seen_at=now
    )
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=["article_id", "feed_code"], set_={"last_seen_at": now}
        )
    )
    return outcome


async def finish_success(
    factory: async_sessionmaker,
    claim: FeedClaim,
    config: FeedConfig,
    result: FetchResult,
    parsed: ParsedFeed | None,
) -> dict:
    async with factory.begin() as session:
        # Serialize identity resolution only within this publisher, including multiple feeds.
        await session.scalar(
            select(NewsSource).where(NewsSource.code == config.code).with_for_update()
        )
        feed = await owned_feed(session, claim)
        now = datetime.now(UTC)
        counts = {"new": 0, "updated": 0, "unchanged": 0, "errors": 0}
        if parsed is not None:
            for error in parsed.errors:
                session.add(
                    RssItemError(
                        run_id=claim.run_id,
                        item_index=error.index,
                        reason=error.reason,
                        item=error.item,
                    )
                )
                counts["errors"] += 1
            for index, article in enumerate(parsed.articles):
                try:
                    outcome = await save_article(session, article, config, now)
                    counts[outcome] += 1
                except ValueError as exc:
                    session.add(
                        RssItemError(
                            run_id=claim.run_id,
                            item_index=index,
                            reason=str(exc),
                            item={"title": article.title, "link": article.original_url},
                        )
                    )
                    counts["errors"] += 1
        run = await session.get(RssRun, claim.run_id)
        run.status = "partial" if counts["errors"] else "success"
        run.http_status = result.status
        run.counts = counts
        run.finished_at = now
        if result.status == 200:
            feed.etag = result.etag
            feed.last_modified = result.last_modified
            feed.last_body_hash = feed.pending_body_hash
        else:
            feed.etag = result.etag or feed.etag
            feed.last_modified = result.last_modified or feed.last_modified
        feed.pending_body_hash = None
        feed.pending_etag = None
        feed.pending_last_modified = None
        feed.last_checked_at = now
        feed.last_success_at = now
        if counts["new"]:
            feed.last_new_at = now
        feed.last_error = None
        feed.consecutive_failures = 0
        feed.next_poll_at = now + timedelta(seconds=config.interval * random.uniform(0.9, 1.1))
        feed.lease_token = None
        feed.lease_until = None
        return counts


async def finish_failure(
    factory: async_sessionmaker,
    claim: FeedClaim,
    config: FeedConfig,
    error: str,
    status: int | None = None,
    retry_after: int = 0,
) -> None:
    async with factory.begin() as session:
        feed = await owned_feed(session, claim)
        now = datetime.now(UTC)
        feed.consecutive_failures += 1
        delay = min(3600, config.interval * 2 ** min(feed.consecutive_failures, 6))
        if status in {401, 403, 404, 410}:
            delay = max(delay, 3600)
        feed.next_poll_at = now + timedelta(
            seconds=max(retry_after, delay * random.uniform(1, 1.1))
        )
        feed.last_checked_at = now
        feed.last_error = error[:500]
        feed.pending_body_hash = None
        feed.pending_etag = None
        feed.pending_last_modified = None
        feed.lease_token = None
        feed.lease_until = None
        run = await session.get(RssRun, claim.run_id)
        run.status = "error"
        run.http_status = status
        run.error = error[:500]
        run.finished_at = now


async def cleanup_history(factory: async_sessionmaker, settings: Settings) -> None:
    async with factory.begin() as session:
        now = datetime.now(UTC)
        pending = select(RssFeed.pending_body_hash).where(RssFeed.pending_body_hash.is_not(None))
        await session.execute(
            delete(RssSnapshot).where(
                RssSnapshot.created_at < now - timedelta(days=settings.rss_snapshot_retention_days),
                RssSnapshot.body_hash.not_in(pending),
            )
        )
        await session.execute(
            delete(RssRun).where(
                RssRun.started_at < now - timedelta(days=settings.rss_run_retention_days),
                RssRun.status != "running",
            )
        )
