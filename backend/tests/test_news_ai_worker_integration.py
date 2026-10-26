"""Focused lease/retry/invalidation checks against a migrated pgvector test DB."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.models.news import NewsArticle, NewsSource
from app.models.news_embeddings import NewsEmbedding
from app.repositories.news_embeddings import (
    NewsLeaseLostError,
    claim_news_batch,
    load_ready_news_embeddings,
    mark_news_claim_failed,
    store_news_claim,
)
from app.services.llm.config import ServicesSettings
from app.services.llm.provider import MockProvider
from app.services.llm.schemas import NewsAnalysis, ProcessedNews


class FakeEmbedder:
    model_version = "bge-m3:1024d"


def _processed(text: str) -> ProcessedNews:
    analysis = NewsAnalysis(summary=f"AI: {text}", impact_kinds=["other"])
    return ProcessedNews(
        url="https://example.test/news",
        text=text,
        summary=analysis.summary,
        markup=analysis,
        vector=[1.0] + [0.0] * 1023,
        embedding_model_version=FakeEmbedder.model_version,
    )


@pytest.fixture
def ai_database():
    database = os.environ.get("LLM_TEST_POSTGRES_DB")
    if not database:
        pytest.skip("Set LLM_TEST_POSTGRES_DB to a dedicated migrated PostgreSQL database")
    settings = Settings(postgres_db=database)
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    created_articles = []
    created_sources = []

    async def add_article(title: str) -> NewsArticle:
        source = NewsSource(
            code=f"aiworker-{uuid4().hex}",
            name="AI worker test",
            site_url="https://example.test",
            kind="media",
        )
        now = datetime.now(UTC)
        article = NewsArticle(
            id=uuid4(),
            source_code=source.code,
            original_url=f"https://example.test/{uuid4().hex}",
            canonical_url=f"https://example.test/canonical/{uuid4().hex}",
            title=title,
            summary=f"RSS {title}",
            content_text=f"Body {title}",
            authors=[],
            published_at=now,
            first_seen_at=now,
            last_seen_at=now,
            updated_at=now,
            source_categories=[],
            topics=["other"],
            content_hash=f"hash-{uuid4().hex}",
            normalizer_version=1,
        )
        async with factory() as session:
            session.add(source)
            await session.flush()
            session.add(article)
            await session.commit()
        created_articles.append(article.id)
        created_sources.append(source.code)
        return article

    yield factory, add_article

    async def cleanup():
        async with factory() as session:
            if created_articles:
                await session.execute(
                    delete(NewsArticle).where(NewsArticle.id.in_(created_articles))
                )
            if created_sources:
                await session.execute(
                    delete(NewsSource).where(NewsSource.code.in_(created_sources))
                )
            await session.commit()
        await engine.dispose()

    asyncio.run(cleanup())


def _settings(**updates) -> ServicesSettings:
    return ServicesSettings(
        _env_file=None,
        llm_mock=True,
        news_worker_batch_size=4,
        news_worker_concurrency=1,
        news_worker_retry_seconds=300,
        **updates,
    )


async def _claim(factory, settings, llm, embedder, *articles):
    return await claim_news_batch(
        factory,
        settings,
        llm=llm,
        embedder=embedder,
        article_ids=[article.id for article in articles],
    )


def test_concurrent_claims_are_disjoint(ai_database):
    factory, add_article = ai_database

    async def scenario():
        first, second = await add_article("one"), await add_article("two")
        settings = _settings()
        llm, embedder = MockProvider(), FakeEmbedder()
        left, right = await asyncio.gather(
            _claim(factory, settings, llm, embedder, first, second),
            _claim(factory, settings, llm, embedder, first, second),
        )
        claimed = [claim.article_id for claim in left + right]
        assert set(claimed) == {first.id, second.id}
        assert len(claimed) == len(set(claimed))

    asyncio.run(scenario())


def test_retry_deadline_terminal_failure_and_changed_input(ai_database):
    factory, add_article = ai_database

    async def scenario():
        article = await add_article("retry")
        settings = _settings(news_worker_max_attempts=2)
        llm, embedder = MockProvider(), FakeEmbedder()
        claim = (await _claim(factory, settings, llm, embedder, article))[0]
        await mark_news_claim_failed(factory, claim, "provider down", settings)
        assert await _claim(factory, settings, llm, embedder, article) == []

        async with factory() as session:
            row = await session.get(NewsEmbedding, article.id)
            assert row.status == "retry" and row.next_attempt_at > datetime.now(UTC)
            row.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
            await session.commit()
        second = (await _claim(factory, settings, llm, embedder, article))[0]
        assert second.attempt == 2
        await mark_news_claim_failed(factory, second, "still down", settings)
        assert await _claim(factory, settings, llm, embedder, article) == []
        async with factory() as session:
            row = await session.get(NewsEmbedding, article.id)
            assert row.status == "failed" and row.next_attempt_at is None

        async with factory() as session:
            current = await session.get(NewsArticle, article.id)
            current.content_text = "changed body"
            current.content_hash = f"changed-{uuid4().hex}"
            await session.commit()
        changed = await _claim(factory, settings, llm, embedder, article)
        assert [item.article_id for item in changed] == [article.id]
        assert changed[0].attempt == 1

    asyncio.run(scenario())


def test_expired_lease_recovers_and_old_owner_cannot_complete(ai_database):
    factory, add_article = ai_database

    async def scenario():
        article = await add_article("crash")
        settings = _settings()
        llm, embedder = MockProvider(), FakeEmbedder()
        old = (await _claim(factory, settings, llm, embedder, article))[0]
        assert await _claim(factory, settings, llm, embedder, article) == []
        async with factory() as session:
            await session.execute(
                update(NewsEmbedding)
                .where(NewsEmbedding.article_id == article.id)
                .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
            )
            await session.commit()
        recovered = (await _claim(factory, settings, llm, embedder, article))[0]
        assert recovered.article_id == old.article_id
        assert recovered.lease_token != old.lease_token
        with pytest.raises(NewsLeaseLostError):
            await store_news_claim(factory, old, _processed(old.text), settings, llm)

    asyncio.run(scenario())


def test_ready_result_is_invalidated_and_stale_claim_cannot_write(ai_database):
    factory, add_article = ai_database

    async def scenario():
        article = await add_article("invalidate")
        settings = _settings()
        llm, embedder = MockProvider(), FakeEmbedder()
        claim = (await _claim(factory, settings, llm, embedder, article))[0]
        await store_news_claim(factory, claim, _processed(claim.text), settings, llm)
        async with factory() as session:
            current = await session.get(NewsArticle, article.id)
            assert article.id in await load_ready_news_embeddings(session, [current], settings)
            current.content_text = "new source text"
            current.content_hash = f"new-{uuid4().hex}"
            await session.commit()
        async with factory() as session:
            current = await session.get(NewsArticle, article.id)
            assert article.id not in await load_ready_news_embeddings(session, [current], settings)

        refreshed = await _claim(factory, settings, llm, embedder, article)
        assert [item.article_id for item in refreshed] == [article.id]
        # A source change does not let another worker steal an active lease.
        async with factory() as session:
            current = await session.get(NewsArticle, article.id)
            current.content_text = "changed again while processing"
            current.content_hash = f"newer-{uuid4().hex}"
            await session.commit()
        assert await _claim(factory, settings, llm, embedder, article) == []

        # The old owner cannot publish a result for the superseded snapshot;
        # after it releases the lease, the changed input becomes claimable.
        await store_news_claim(
            factory, refreshed[0], _processed(refreshed[0].text), settings, llm
        )
        newest = await _claim(factory, settings, llm, embedder, article)
        assert [item.article_id for item in newest] == [article.id]
        async with factory() as session:
            row = await session.scalar(
                select(NewsEmbedding).where(NewsEmbedding.article_id == article.id)
            )
        assert row.status == "processing"

    asyncio.run(scenario())
