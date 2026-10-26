"""Isolated schema test; opt in with ARTICLE_CONTENT_TEST_POSTGRES_DB."""

import asyncio
import os
import runpy
import zlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.base import Base
from app.models.news import (
    NewsArticle,
    NewsArticleFeed,
    NewsArticleKey,
    NewsSource,
    RssFeed,
    RssSnapshot,
)
from app.repositories.article_content import claim_article_content, finish_article_content
from app.repositories.news import save_article, sync_catalog
from app.services.rss.backfill import backfill_rss
from app.services.rss.catalog import load_catalog
from app.services.rss.parser import parse_feed
from tests.test_article_content import BODY
from tests.test_rss import item, rss


def test_leases_retry_hash_invalidation_and_rss_body_preservation():
    database = os.environ.get("ARTICLE_CONTENT_TEST_POSTGRES_DB")
    if not database:
        pytest.skip("Set ARTICLE_CONTENT_TEST_POSTGRES_DB for isolated-schema PostgreSQL test")

    async def scenario():
        settings = Settings(postgres_db=database)
        engine = create_async_engine(settings.database_url)
        schema = "content_test_" + uuid4().hex
        tables = [m.__table__ for m in (
            NewsSource, RssFeed, NewsArticle, NewsArticleKey, NewsArticleFeed, RssSnapshot,
        )]
        try:
            async with engine.begin() as connection:
                await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
                translated = await connection.execution_options(schema_translate_map={None: schema})
                await translated.run_sync(
                    lambda bind: Base.metadata.create_all(bind, tables=tables)
                )
            factory = async_sessionmaker(
                engine.execution_options(schema_translate_map={None: schema}),
                expire_on_commit=False,
            )
            config = next(c for c in load_catalog() if c.code == "bfm")
            await sync_catalog(factory, (config,))
            incoming = parse_feed(rss(item(link="https://www.bfm.ru/news/1")), config).articles[0]
            async with factory.begin() as session:
                await save_article(session, incoming, config, datetime.now(UTC))
                original_hash = await session.scalar(select(NewsArticle.content_hash))
            claim = await claim_article_content(factory, [config.code])
            assert claim is not None
            assert await claim_article_content(factory, [config.code]) is None
            # Expired tokens must not publish a stale body, even before a new owner claims it.
            async with factory.begin() as session:
                await session.execute(update(NewsArticle).values(
                    content_lease_until=datetime.now(UTC) - timedelta(seconds=1)
                ))
            assert not await finish_article_content(factory, claim, text=BODY)
            retry_claim = await claim_article_content(factory, [config.code])
            assert retry_claim.token != claim.token
            assert await finish_article_content(factory, retry_claim, error="HTTP 503")
            async with factory.begin() as session:
                article = await session.scalar(select(NewsArticle))
                assert article.content_status == "retry" and article.content_error == "HTTP 503"
                article.content_next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
            ready_claim = await claim_article_content(factory, [config.code])
            assert await finish_article_content(factory, ready_claim, text=BODY)
            async with factory.begin() as session:
                article = await session.scalar(select(NewsArticle))
                assert article.content_hash != original_hash
                ready_hash = article.content_hash
                assert article.content_origin == "article" and article.content_status == "ready"
                # The next RSS poll must not erase the downloaded body or change its hash.
                await save_article(session, incoming, config, datetime.now(UTC))
            async with factory() as session:
                article = await session.scalar(select(NewsArticle))
                assert article.content_text == BODY and article.content_hash == ready_hash
                assert article.content_status == "ready" and article.content_error is None
            assert await claim_article_content(factory, [config.code]) is None
            # A new editorial revision queues a refetch; a later RSS body revokes that lease.
            revision = replace(incoming, source_updated_at=datetime.now(UTC))
            async with factory.begin() as session:
                await save_article(session, revision, config, datetime.now(UTC))
            stale_claim = await claim_article_content(factory, [config.code])
            assert stale_claim is not None
            async with factory.begin() as session:
                await save_article(session, replace(revision, content_text=BODY + " RSS."),
                                   config, datetime.now(UTC))
            assert not await finish_article_content(factory, stale_claim, text=BODY)
            # Recover the old RBC caption case from stored RSS, preserving title/metadata.
            rbc = next(c for c in load_catalog() if c.code == "rbc")
            await sync_catalog(factory, (rbc,))
            rbc_body = rss(item(link="https://www.rbc.ru/news/1", extra=(
                f"<rbc:full-text>{BODY}</rbc:full-text>"
            )))
            rbc_article = parse_feed(rbc_body, rbc).articles[0]
            async with factory.begin() as session:
                await save_article(session, replace(rbc_article, content_text=None),
                                   rbc, datetime.now(UTC))
                row = await session.scalar(select(NewsArticle).where(
                    NewsArticle.source_code == "rbc"
                ))
                row.content_text = "Джефф Безос"
                session.add(RssSnapshot(feed_code=rbc.feed_code, body_hash=uuid4().hex,
                                        compressed_body=zlib.compress(rbc_body)))
            assert await backfill_rss(factory) == 1
            assert await backfill_rss(factory) == 0
            async with factory() as session:
                row = await session.scalar(select(NewsArticle).where(
                    NewsArticle.source_code == "rbc"
                ))
                assert row.content_text == BODY and row.content_origin == "rss"
        finally:
            async with engine.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            await engine.dispose()
    asyncio.run(scenario())


def test_article_content_migration_preserves_existing_text():
    database = os.environ.get("ARTICLE_CONTENT_TEST_POSTGRES_DB")
    if not database:
        pytest.skip("Set ARTICLE_CONTENT_TEST_POSTGRES_DB for isolated-schema PostgreSQL test")
    migration = runpy.run_path(str(
        Path(__file__).parents[1] / "alembic/versions/0010_article_content.py"
    ))

    def migrate(connection, direction):
        with Operations.context(MigrationContext.configure(connection)):
            migration[direction]()

    async def scenario():
        engine = create_async_engine(Settings(postgres_db=database).database_url)
        schema = "content_migration_test_" + uuid4().hex
        try:
            async with engine.begin() as connection:
                await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
                await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
                await connection.execute(text(
                    "CREATE TABLE news_articles (id integer PRIMARY KEY, content_text text)"
                ))
                await connection.execute(text("INSERT INTO news_articles VALUES (1, 'original')"))
                await connection.run_sync(migrate, "upgrade")
                assert (await connection.execute(text(
                    "SELECT content_text, content_status FROM news_articles"
                ))).one() == ("original", "pending")
                await connection.run_sync(migrate, "downgrade")
                assert await connection.scalar(text(
                    "SELECT content_text FROM news_articles"
                )) == "original"
        finally:
            async with engine.begin() as connection:
                await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            await engine.dispose()
    asyncio.run(scenario())
