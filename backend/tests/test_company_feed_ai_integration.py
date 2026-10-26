import asyncio
import os

import pytest
from sqlalchemy import update

from app.core.config import Settings
from app.models.company import Company
from app.models.company_profile import CompanyProfileCache
from app.models.news import NewsArticle
from app.models.news_embeddings import NewsEmbedding
from app.repositories.company_profiles import prepare_company_profile_input
from app.repositories.news_embeddings import (
    active_embedding_model_version,
    llm_model_version,
    news_input_fingerprint,
)
from app.services.llm.config import ServicesSettings
from app.services.llm.news_analysis import NEWS_PROMPT_VERSION
from tests.test_company_news_integration import api, seed, sessions


@pytest.fixture
def feed_settings():
    name = os.environ.get("COMPANY_NEWS_TEST_POSTGRES_DB")
    if not name:
        pytest.skip("Set COMPANY_NEWS_TEST_POSTGRES_DB to a dedicated migrated database")
    return Settings(postgres_db=name, rss_enabled=False, analytics_enabled=False)


def test_feed_uses_only_current_cached_ai_and_invalidates_changed_inputs(
    feed_settings, monkeypatch
):
    data = seed(feed_settings)
    ai = ServicesSettings(
        _env_file=None,
        llm_mock=True,
        news_worker_enabled=False,
        profile_enabled=True,
    )

    async def store_ready_cache():
        async with sessions(feed_settings) as factory, factory.begin() as session:
            company = await session.get(Company, data.company_id)
            article = await session.get(NewsArticle, data.news_id)
            prepared = await prepare_company_profile_input(session, company, ai)
            vector = [1.0, *([0.0] * 1023)]
            session.add(
                CompanyProfileCache(
                    company_id=company.id,
                    input_fingerprint=prepared.fingerprint,
                    profile_text=prepared.text.model_dump(mode="json"),
                    fields=prepared.fields.model_dump(mode="json"),
                    vectors={key: vector for key in ["main", *prepared.text.aspects]},
                    embedding_model_version=active_embedding_model_version(ai),
                    status="ready",
                    attempts=0,
                )
            )
            session.add(
                NewsEmbedding(
                    article_id=article.id,
                    summary="Краткое резюме из готового кеша",
                    vector=vector,
                    analysis={
                        "topics": ["business"],
                        "affected_sectors": [{"code": "62.01", "name": "Разработка ПО"}],
                        "regions": ["Татарстан"],
                        "impact_kinds": [],
                        "facts": [],
                        "size_relevance": [],
                    },
                    source_content_hash=article.content_hash,
                    input_fingerprint=news_input_fingerprint(article, ai),
                    llm_model_version=llm_model_version(ai),
                    prompt_version=NEWS_PROMPT_VERSION,
                    embedding_model_version=active_embedding_model_version(ai),
                    status="ready",
                    attempts=0,
                )
            )

    asyncio.run(store_ready_cache())
    monkeypatch.setattr(
        "app.services.llm.embeddings.EmbeddingsClient.embed",
        lambda *args, **kwargs: pytest.fail("GET must not call Ollama"),
    )
    monkeypatch.setattr(
        "app.repositories.news_embeddings.get_llm_provider",
        lambda *args, **kwargs: pytest.fail("GET must not create an LLM provider"),
    )

    with api(feed_settings, data, ai) as client:
        url = f"/companies/{data.company_id}/news"
        ready = client.get(url).json()
        assert ready["ranking_method"] == "hybrid"
        assert ready["personalization_status"] == "ready"
        assert ready["items"][0]["summary_origin"] == "ai"
        assert ready["items"][0]["summary"] == "Краткое резюме из готового кеша"
        assert ready["items"][0]["processing_status"] == "ready"

        async def make_news_model_stale():
            async with sessions(feed_settings) as factory, factory.begin() as session:
                await session.execute(
                    update(NewsEmbedding)
                    .where(NewsEmbedding.article_id == data.news_id)
                    .values(embedding_model_version="old-model:1024d")
                )

        asyncio.run(make_news_model_stale())
        stale_news = client.get(url).json()
        assert stale_news["items"][0]["summary_origin"] == "rss"
        assert stale_news["items"][0]["processing_status"] == "unavailable"

        async def change_company_input():
            async with sessions(feed_settings) as factory, factory.begin() as session:
                await session.execute(
                    update(Company).where(Company.id == data.company_id).values(city="Иннополис")
                )

        asyncio.run(change_company_input())
        stale_profile = client.get(url).json()
        assert stale_profile["ranking_method"] == "rules"
        assert stale_profile["personalization_status"] == "preparing"
