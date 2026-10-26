"""Set LLM_TEST_POSTGRES_DB to a dedicated migrated PostgreSQL database."""

import asyncio
import hashlib
import math
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.models.news import NewsArticle, NewsSource
from app.models.news_embeddings import NewsEmbedding
from app.repositories.news_embeddings import (
    fetch_news_events,
    fetch_news_records,
    process_and_store_news,
    rank_relevant_for_company,
)
from app.services.llm.company_profile import build_company_profile
from app.services.llm.config import ServicesSettings
from app.services.llm.provider import MockProvider
from app.services.llm.schemas import CompanyFields

DIM = 1024


class FakeEmbedder:
    model_version = "fake-embedder:1024d"

    def embed(self, texts):
        return [_vector(text) for text in texts]

    def embed_one(self, text):
        return _vector(text)


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [digest[i % len(digest)] / 255.0 for i in range(DIM)]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


@pytest.fixture
def run_in():
    database = os.environ.get("LLM_TEST_POSTGRES_DB")
    if not database:
        pytest.skip("Set LLM_TEST_POSTGRES_DB to a dedicated migrated PostgreSQL database")
    engines = []

    def run(coroutine_factory):
        engine = create_async_engine(
            Settings(postgres_db=database).database_url, poolclass=NullPool
        )
        engines.append(engine)
        factory = async_sessionmaker(engine, expire_on_commit=False)

        async def runner():
            async with factory() as session:
                return await coroutine_factory(session)

        return asyncio.run(runner())

    yield run
    for engine in engines:
        asyncio.run(engine.dispose())


async def _make_row(session):
    source = NewsSource(
        code=f"llmtest{uuid4().hex[:10]}",
        name="LLM test source",
        site_url="https://example.ru",
        kind="media",
    )
    session.add(source)
    await session.flush()  # источник должен существовать до вставки статьи (FK)
    link = f"https://example.ru/news-{uuid4().hex[:8]}"
    article = NewsArticle(
        id=uuid4(),
        source_code=source.code,
        original_url=link,
        canonical_url=link,
        title="Банк снизил ключевую ставку",
        summary="RSS-анонс ставки",
        content_text="Ключевая ставка снижена на 50 б.п., кредиты для бизнеса станут дешевле.",
        authors=[],
        published_at=datetime.now(UTC),
        first_seen_at=datetime.now(UTC),
        last_seen_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        source_categories=[],
        topics=["other"],
        content_hash=uuid4().hex,
        normalizer_version=1,
    )
    session.add(article)
    return article


def test_roundtrip_process_fetch_and_rank(run_in):
    async def scenario(session):
        article = await _make_row(session)
        await session.commit()

        llm, embedder = MockProvider(), FakeEmbedder()
        ai_settings = ServicesSettings(
            _env_file=None, llm_mock=True, embed_model="fake-embedder", embed_dim=DIM
        )
        outcome = await process_and_store_news(
            session, [article.id], llm=llm, embedder=embedder, settings=ai_settings
        )
        row = await session.scalar(
            select(NewsEmbedding).where(NewsEmbedding.article_id == article.id)
        )

        again = await process_and_store_news(
            session, [article.id], llm=llm, embedder=embedder, settings=ai_settings
        )
        forced = await process_and_store_news(
            session,
            [article.id],
            llm=llm,
            embedder=embedder,
            settings=ai_settings,
            force=True,
        )

        records = await fetch_news_records(session, [article.id], settings=ai_settings)
        events = await fetch_news_events(session, [article.id], settings=ai_settings)

        fields = CompanyFields(synthetic_u="U_TEST")
        profile = build_company_profile(fields, embedder=FakeEmbedder())
        ranked = await rank_relevant_for_company(
            session, profile, fields, article_ids=[article.id], settings=ai_settings
        )
        filtered = await rank_relevant_for_company(
            session,
            profile,
            fields,
            article_ids=[article.id],
            min_score=2.0,
            settings=ai_settings,
        )
        return article, outcome, row, again, forced, records, events, ranked, filtered

    article, outcome, row, again, forced, records, events, ranked, filtered = run_in(scenario)

    assert outcome.missing == [] and outcome.skipped == []
    processed = outcome.processed[article.id]
    assert len(processed.vector) == DIM
    assert processed.markup.topics == ["демо"]
    assert row.embedding_model_version == "fake-embedder:1024d"
    assert "summary" not in row.analysis

    record = records[0]
    assert record.summary == processed.summary
    assert record.vector == pytest.approx(processed.vector, abs=1e-6)  # pgvector хранит float32
    assert record.markup.summary == processed.summary
    assert record.markup.impact_kinds == ["other"]

    assert events[0].title == article.title
    assert events[0].url == article.original_url

    assert again.skipped == [article.id]
    assert again.processed == {}
    assert forced.skipped == []
    assert article.id in forced.processed
    assert row.updated_at is not None

    assert len(ranked) == 1
    assert ranked[0].company_id == "U_TEST"
    assert ranked[0].news_url == article.original_url
    assert ranked[0].score >= 0
    assert filtered == []


def test_process_and_store_reports_missing_ids(run_in):
    async def scenario(session):
        ghost = uuid4()
        outcome = await process_and_store_news(session, [ghost], llm=MockProvider())
        return ghost, outcome

    ghost, outcome = run_in(scenario)
    assert outcome.missing == [ghost]
    assert outcome.processed == {}
