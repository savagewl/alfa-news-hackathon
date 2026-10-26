"""Set RSS_TEST_POSTGRES_DB to a dedicated database with all migrations applied."""

import asyncio
import os
import zlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.dependencies import get_current_user
from app.core.config import Settings
from app.main import create_app
from app.models.news import (
    NewsArticle,
    NewsArticleFeed,
    NewsArticleKey,
    RssFeed,
    RssItemError,
    RssRun,
    RssSnapshot,
)
from app.repositories.news import (
    LeaseLostError,
    claim_feed,
    cleanup_history,
    finish_failure,
    finish_success,
    save_snapshot,
    sync_catalog,
)
from app.services.rss.catalog import load_catalog
from app.services.rss.fetcher import FetchResult
from app.services.rss.parser import parse_feed
from app.services.rss.worker import RssWorker
from tests.test_rss import item, rss


@pytest.fixture
def settings():
    database = os.environ.get("RSS_TEST_POSTGRES_DB")
    if not database:
        pytest.skip("Set RSS_TEST_POSTGRES_DB to a dedicated migrated PostgreSQL database")
    return Settings(postgres_db=database, rss_enabled=False)


@asynccontextmanager
async def sessions(settings):
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


def config():
    code = "test-" + uuid4().hex
    return load_catalog()[0].model_copy(update={"code": code, "feed_code": code + "-feed"})


async def due(factory, cfg):
    async with factory.begin() as session:
        await session.execute(
            update(RssFeed)
            .where(RssFeed.code == cfg.feed_code)
            .values(next_poll_at=datetime.now(UTC) - timedelta(seconds=1))
        )


async def ingest(factory, cfg, body):
    await due(factory, cfg)
    claim = await claim_feed(factory, (cfg,), 120)
    assert claim is not None
    result = FetchResult(200, body, '"test-etag"', "Wed, 21 Oct 2026 12:00:00 GMT")
    await save_snapshot(factory, claim, result)
    return await finish_success(factory, claim, cfg, result, parse_feed(body, cfg))


def test_idempotency_updates_aliases_and_quarantine(settings):
    async def scenario():
        async with sessions(settings) as factory:
            cfg = config()
            await sync_catalog(factory, (cfg,))
            body = rss(item(extra="<guid>stable</guid><description>Summary</description>"))
            first = await ingest(factory, cfg, body)
            assert first == {"new": 1, "updated": 0, "unchanged": 0, "errors": 0}
            second = await ingest(factory, cfg, body)
            assert second["unchanged"] == 1 and second["new"] == 0
            changed = rss(
                item("Changed", "https://example.com/new", "<guid>stable</guid>")
                + "<item><title>Bad</title></item>"
            )
            third = await ingest(factory, cfg, changed)
            assert third["updated"] == 1 and third["errors"] == 1
            async with factory() as session:
                articles = (
                    await session.scalars(
                        select(NewsArticle).where(NewsArticle.source_code == cfg.code)
                    )
                ).all()
                assert len(articles) == 1
                assert articles[0].title == "Changed"
                assert articles[0].summary == "Summary"
                assert articles[0].canonical_url == "https://example.com/new"
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(NewsArticleKey)
                        .where(NewsArticleKey.source_code == cfg.code)
                    )
                    == 3
                )
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(RssSnapshot)
                        .where(RssSnapshot.feed_code == cfg.feed_code)
                    )
                    == 2
                )
                errors = (
                    await session.scalars(
                        select(RssItemError).join(RssRun).where(RssRun.feed_code == cfg.feed_code)
                    )
                ).all()
                assert len(errors) == 1
                feed = await session.get(RssFeed, cfg.feed_code)
                assert feed.etag == '"test-etag"'
                assert feed.pending_body_hash is None

    asyncio.run(scenario())


def test_claim_race_expired_lease_and_stale_writer(settings):
    async def scenario():
        async with sessions(settings) as factory:
            cfg = config()
            await sync_catalog(factory, (cfg,))
            claims = await asyncio.gather(*(claim_feed(factory, (cfg,), 120) for _ in range(8)))
            owned = [c for c in claims if c is not None]
            assert len(owned) == 1
            original = owned[0]
            async with factory.begin() as session:
                await session.execute(
                    update(RssFeed)
                    .where(RssFeed.code == cfg.feed_code)
                    .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
                )
            reclaimed = await claim_feed(factory, (cfg,), 120)
            assert reclaimed is not None and reclaimed.token != original.token
            with pytest.raises(LeaseLostError):
                await finish_success(
                    factory, original, cfg, FetchResult(304, b"", None, None), None
                )
            await finish_success(factory, reclaimed, cfg, FetchResult(304, b"", None, None), None)
            async with factory() as session:
                assert (await session.get(RssRun, original.run_id)).status == "abandoned"

    asyncio.run(scenario())


def test_transaction_rollback_and_pending_snapshot_recovery(settings, monkeypatch):
    import app.repositories.news as repository

    async def scenario():
        async with sessions(settings) as factory:
            cfg = config()
            await sync_catalog(factory, (cfg,))
            claim = await claim_feed(factory, (cfg,), 120)
            body = rss(item("A", "https://example.com/a") + item("B", "https://example.com/b"))
            result = FetchResult(200, body, '"pending"', None)
            await save_snapshot(factory, claim, result)
            original_save = repository.save_article
            calls = 0

            async def fail_second(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise SQLAlchemyError("Simulated persistence failure")
                return await original_save(*args)

            with monkeypatch.context() as patch:
                patch.setattr(repository, "save_article", fail_second)
                with pytest.raises(SQLAlchemyError):
                    await finish_success(factory, claim, cfg, result, parse_feed(body, cfg))
            async with factory.begin() as session:
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(NewsArticle)
                        .where(NewsArticle.source_code == cfg.code)
                    )
                    == 0
                )
                feed = await session.get(RssFeed, cfg.feed_code)
                assert feed.etag is None
                assert feed.pending_body_hash is not None
                feed.lease_until = datetime.now(UTC) - timedelta(seconds=1)
            recovered = await claim_feed(factory, (cfg,), 120)
            assert zlib.decompress(recovered.pending_body) == body
            worker = RssWorker(factory, settings)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda request: pytest.fail("Recovery must use the saved snapshot")
                )
            ) as client:
                await worker.process(client, recovered, cfg)
            async with factory() as session:
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(NewsArticle)
                        .where(NewsArticle.source_code == cfg.code)
                    )
                    == 2
                )
                feed = await session.get(RssFeed, cfg.feed_code)
                assert feed.etag == '"pending"'
                assert feed.pending_body_hash is None

    asyncio.run(scenario())


def test_multiple_feeds_concurrent_dedup_and_distinct_publishers(settings):
    async def scenario():
        async with sessions(settings) as factory:
            first = config()
            second = first.model_copy(update={"feed_code": first.feed_code + "-second"})
            other = config()
            await sync_catalog(factory, (first, second, other))
            body = rss(item(extra="<guid>same</guid>"))
            counts = await asyncio.gather(
                ingest(factory, first, body),
                ingest(factory, second, body),
                ingest(factory, other, body),
            )
            assert sum(c["new"] for c in counts) == 2
            async with factory() as session:
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(NewsArticle)
                        .where(NewsArticle.source_code == first.code)
                    )
                    == 1
                )
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(NewsArticleFeed)
                        .where(NewsArticleFeed.feed_code.in_([first.feed_code, second.feed_code]))
                    )
                    == 2
                )

    asyncio.run(scenario())


def test_failure_backoff_and_retention(settings):
    async def scenario():
        async with sessions(settings) as factory:
            cfg = config()
            await sync_catalog(factory, (cfg,))
            claim = await claim_feed(factory, (cfg,), 120)
            body = rss(item())
            await save_snapshot(factory, claim, FetchResult(200, body, '"unused"', None))
            await finish_failure(factory, claim, cfg, "Feed returned HTTP 429", 429, 7200)
            async with factory.begin() as session:
                feed = await session.get(RssFeed, cfg.feed_code)
                assert feed.etag is None and feed.pending_body_hash is None
                assert feed.consecutive_failures == 1
                assert feed.next_poll_at > datetime.now(UTC) + timedelta(seconds=7100)
                await session.execute(
                    update(RssSnapshot)
                    .where(RssSnapshot.feed_code == cfg.feed_code)
                    .values(created_at=datetime.now(UTC) - timedelta(days=8))
                )
                run = await session.get(RssRun, claim.run_id)
                run.started_at = datetime.now(UTC) - timedelta(days=31)
            await cleanup_history(factory, settings)
            async with factory() as session:
                assert await session.get(RssRun, claim.run_id) is None
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(RssSnapshot)
                        .where(RssSnapshot.feed_code == cfg.feed_code)
                    )
                    == 0
                )

    asyncio.run(scenario())


def test_news_api_filters_cursor_and_missing_article(settings):
    cfg = config()

    async def seed():
        async with sessions(settings) as factory:
            await sync_catalog(factory, (cfg,))
            body = rss(
                "".join(
                    item(
                        str(i),
                        f"https://example.com/{i}",
                        "<category>Экономика</category>"
                        "<pubDate>Wed, 21 Oct 2026 12:00:00 +0300</pubDate>",
                    )
                    for i in range(3)
                )
            )
            await ingest(factory, cfg, body)

    asyncio.run(seed())
    app = create_app(settings)
    app.dependency_overrides[get_current_user] = lambda: object()
    with TestClient(app) as client:
        params = {"source": cfg.code, "limit": 2, "topic": "economy"}
        response = client.get("/news", params=params)
        assert response.status_code == 200, response.text
        page = response.json()
        assert len(page["items"]) == 2 and page["next_cursor"]
        params["cursor"] = page["next_cursor"]
        second = client.get("/news", params=params).json()
        assert len(second["items"]) == 1 and second["next_cursor"] is None
        assert len({a["id"] for a in page["items"] + second["items"]}) == 3
        assert client.get("/news/" + page["items"][0]["id"]).status_code == 200
        assert client.get("/news/" + str(uuid4())).status_code == 404
        assert client.get("/news?cursor=bad").status_code == 422
        assert client.get("/news?limit=101").status_code == 422
        assert client.get("/news?seen_from=2026-10-21").status_code == 422
        period = client.get(
            "/news", params={"source": cfg.code, "published_from": "2026-10-22T00:00:00Z"}
        ).json()
        assert period["items"] == []
        sources = client.get("/news/sources")
        assert sources.status_code == 200
        status = next(s for s in sources.json() if s["code"] == cfg.code)
        assert status["feeds"][0]["last_success_at"]


def test_worker_parallelism_and_shutdown_keep_api_responsive(settings, monkeypatch):
    configs = tuple(config() for _ in range(5))
    active, maximum, done = 0, 0, set()

    async def fake_fetch(client, cfg, *args):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.2)
            done.add(cfg.feed_code)
            return FetchResult(200, rss(item()), None, None)
        finally:
            active -= 1

    monkeypatch.setattr("app.services.rss.worker.load_catalog", lambda: configs)
    monkeypatch.setattr("app.services.rss.worker.fetch_feed", fake_fetch)
    # Catalog URLs use one domain; use different allowed hosts to exercise parallel fetches.
    configs = tuple(
        c.model_copy(update={"url": f"https://host-{i}.test/rss"}) for i, c in enumerate(configs)
    )
    app = create_app(settings.model_copy(update={"rss_enabled": True}))
    app.dependency_overrides[get_current_user] = lambda: object()
    with TestClient(app) as client:
        import time

        deadline = time.monotonic() + 15
        while len(done) < 5 and time.monotonic() < deadline:
            assert client.get("/health").status_code == 200
            time.sleep(0.05)
        assert len(done) == 5
        assert maximum == settings.rss_concurrency
    assert app.state.rss_worker.task.done()
