import asyncio
import json
import os
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, select, update

from app.core.config import Settings
from app.models.analytics import AnalyticsChunk, AnalyticsDocument
from app.models.company import Company
from app.models.news_assessment import NewsAssessment
from app.repositories.assessments import claim_assessment
from app.services.assessments import AssessmentWorker
from app.services.llm.config import ServicesSettings
from app.services.llm.provider import MockProvider
from tests.test_company_news_integration import api, seed, sessions


@pytest.fixture
def settings():
    name = os.environ.get("COMPANY_NEWS_TEST_POSTGRES_DB")
    if not name:
        pytest.skip("Set COMPANY_NEWS_TEST_POSTGRES_DB to a dedicated migrated database")
    settings = Settings(postgres_db=name, rss_enabled=False, analytics_enabled=False)

    async def clear_queue():
        # The worker claims rows of any company, so earlier runs must not leak into this one.
        async with sessions(settings) as factory, factory.begin() as session:
            await session.execute(delete(NewsAssessment))

    asyncio.run(clear_queue())
    return settings


ANSWER = {
    "brief_impact": "Рост производства компьютеров повлияет на спрос компании.",
    "risks": "Рост цен на комплектующие сократит маржу. Возможны сбои поставок.",
    "opportunities": "Спрос на компьютеры в регионе вырастет. Это позволит нарастить выпуск.",
    "bank_analytics": "Аналитики банка допускают рост производства компьютеров.",
    "bank_refs": [1],
}


def add_bank_document(settings, data) -> str:
    document_id = uuid4()

    async def insert():
        async with sessions(settings) as factory, factory.begin() as session:
            session.add(
                AnalyticsDocument(
                    id=document_id,
                    sha256=str(document_id),
                    filename="outlook.pdf",
                    title="Прогноз производства компьютеров",
                    publisher="Альфа-Банк",
                    published_at=date(2024, 1, 1),
                    date_origin="provided",
                    uploaded_by=data.user_id,
                    content=b"%PDF-test",
                    status="parsed",
                    page_count=3,
                    chunk_count=2,
                )
            )
            await session.flush()
            session.add(
                AnalyticsChunk(
                    document_id=document_id,
                    chunk_id="p2-1",
                    page_number=2,
                    section="Прогноз",
                    text="Прогноз: производство компьютеров может увеличиться",
                    search_text="Прогноз производство компьютеров",
                    bbox=[],
                )
            )
            session.add(
                AnalyticsChunk(
                    document_id=document_id,
                    chunk_id="p3-1",
                    page_number=3,
                    section="Другая тема",
                    text="Несвязанный раздел отчёта",
                    search_text="Несвязанный раздел отчёта",
                    bbox=[],
                )
            )

    asyncio.run(insert())
    return str(document_id)


def ai_settings(document_id: str, **overrides) -> ServicesSettings:
    values = {
        "llm_api_base": "https://llm.test/v1",
        "bank_document_ids": [document_id],
        "assessment_enabled": True,
        "assessment_retry_seconds": 1,
        **overrides,
    }
    return ServicesSettings(_env_file=None, **values)


def run_worker(settings, ai, handler) -> int:
    async def run() -> int:
        async with sessions(settings) as factory:
            worker = AssessmentWorker(factory, ai, llm=MockProvider(handler))
            return await worker.run_once()

    return asyncio.run(run())


def assessment_rows(settings, data) -> list[NewsAssessment]:
    async def load():
        async with sessions(settings) as factory, factory.begin() as session:
            return list(
                await session.scalars(
                    select(NewsAssessment).where(NewsAssessment.company_id == data.company_id)
                )
            )

    return asyncio.run(load())


def feed(client, data) -> dict:
    result = client.get(f"/companies/{data.company_id}/news", params={"source": data.source})
    assert result.status_code == 200, result.text
    return result.json()["items"][0]


def test_feed_queues_assessment_and_shows_risks_opportunities_and_bank_analytics(settings):
    data = seed(settings)
    document_id = add_bank_document(settings, data)
    ai = ai_settings(document_id)
    prompts = []

    def llm(purpose, messages, json_mode):
        assert purpose == "news_assessment" and json_mode
        prompts.append(messages[-1]["content"])
        return json.dumps(ANSWER, ensure_ascii=False)

    with api(settings, data, ai) as client:
        item = feed(client, data)
        assert item["assessment_status"] == "pending" and item["risks"] is None
        assert [row.status for row in assessment_rows(settings, data)] == ["pending"]
        # Repeated feed requests must not duplicate the queue entry.
        assert feed(client, data)["assessment_status"] == "pending"

        assert run_worker(settings, ai, llm) == 1
        assert len(prompts) == 1 and "[B1]" in prompts[0] and "Прогноз" in prompts[0]

        item = feed(client, data)
    assert item["assessment_status"] == "ready"
    assert item["brief_impact"] == ANSWER["brief_impact"]
    assert item["risks"] == [ANSWER["risks"]]
    assert item["opportunities"] == [ANSWER["opportunities"]]
    assert item["bank_analytics"] == ANSWER["bank_analytics"]
    assert [source["evidence_ref"] for source in item["bank_sources"]] == [
        f"bank:{document_id}#p2-1"
    ]
    assert item["bank_sources"][0]["page_number"] == 2
    assert [row.status for row in assessment_rows(settings, data)] == ["ready"]


def test_bank_analytics_dropped_without_valid_references(settings):
    data = seed(settings)
    ai = ai_settings(add_bank_document(settings, data))
    answer = {**ANSWER, "bank_refs": [7]}

    with api(settings, data, ai) as client:
        feed(client, data)
        run_worker(settings, ai, lambda *a: json.dumps(answer, ensure_ascii=False))
        item = feed(client, data)
    assert item["assessment_status"] == "ready"
    assert item["bank_analytics"] is None and item["bank_sources"] == []


def test_company_change_requeues_assessment(settings):
    data = seed(settings)
    ai = ai_settings(add_bank_document(settings, data))

    async def move_company():
        async with sessions(settings) as factory, factory.begin() as session:
            await session.execute(
                update(Company).where(Company.id == data.company_id).values(city="Казань")
            )

    with api(settings, data, ai) as client:
        feed(client, data)
        run_worker(settings, ai, lambda *a: json.dumps(ANSWER, ensure_ascii=False))
        assert feed(client, data)["assessment_status"] == "ready"
        asyncio.run(move_company())
        assert feed(client, data)["assessment_status"] == "pending"


def test_failed_generation_retries_then_gives_up(settings):
    data = seed(settings)
    ai = ai_settings(add_bank_document(settings, data), assessment_max_attempts=2)

    def broken(*args):
        raise RuntimeError("LLM is down")

    async def make_due():
        async with sessions(settings) as factory, factory.begin() as session:
            await session.execute(
                update(NewsAssessment)
                .where(NewsAssessment.company_id == data.company_id)
                .values(next_attempt_at=datetime.now(UTC) - timedelta(seconds=1))
            )

    with api(settings, data, ai) as client:
        feed(client, data)
        assert run_worker(settings, ai, broken) == 1
        assert [row.status for row in assessment_rows(settings, data)] == ["retry"]
        assert feed(client, data)["assessment_status"] == "pending"
        assert run_worker(settings, ai, broken) == 0  # waits for next_attempt_at
        asyncio.run(make_due())
        assert run_worker(settings, ai, broken) == 1
        row = assessment_rows(settings, data)[0]
        assert row.status == "failed" and "LLM is down" in row.last_error
        # After the final failure the feed neither loops the queue nor fails.
        assert feed(client, data)["assessment_status"] == "not_assessed"


def test_assessments_disabled_or_llm_unconfigured_do_not_queue(settings):
    data = seed(settings)
    document_id = add_bank_document(settings, data)
    unconfigured = ServicesSettings(_env_file=None, assessment_enabled=True)
    disabled = ai_settings(document_id, assessment_enabled=False)
    for ai in (unconfigured, disabled):
        with api(settings, data, ai) as client:
            assert feed(client, data)["assessment_status"] == "not_assessed"
    assert assessment_rows(settings, data) == []


def test_ready_assessment_cannot_be_overwritten_by_a_late_claim(settings):
    data = seed(settings)
    ai = ai_settings(add_bank_document(settings, data))
    with api(settings, data, ai) as client:
        feed(client, data)
        run_worker(settings, ai, lambda *a: json.dumps(ANSWER, ensure_ascii=False))
        stored = assessment_rows(settings, data)[0]

        async def claim():
            async with sessions(settings) as factory, factory() as session:
                return await claim_assessment(
                    session, data.company_id, data.news_id,
                    stored.input_fingerprint, lease_seconds=60,
                )

        assert asyncio.run(claim()) is None
        assert assessment_rows(settings, data)[0].result == stored.result
        assert feed(client, data)["brief_impact"] == ANSWER["brief_impact"]


def test_only_changes_to_used_evidence_invalidate_assessment(settings):
    data = seed(settings)
    document_id = add_bank_document(settings, data)
    ai = ai_settings(document_id)

    async def change_chunk(chunk_id):
        async with sessions(settings) as factory, factory.begin() as session:
            await session.execute(
                update(AnalyticsChunk)
                .where(
                    AnalyticsChunk.document_id == document_id,
                    AnalyticsChunk.chunk_id == chunk_id,
                )
                .values(text="Обновлённый прогноз")
            )

    with api(settings, data, ai) as client:
        feed(client, data)
        run_worker(settings, ai, lambda *a: json.dumps(ANSWER, ensure_ascii=False))
        asyncio.run(change_chunk("p3-1"))
        assert feed(client, data)["assessment_status"] == "ready"
        asyncio.run(change_chunk("p2-1"))
        assert feed(client, data)["assessment_status"] == "pending"
        assert run_worker(settings, ai, lambda *a: json.dumps(ANSWER, ensure_ascii=False)) == 1
        assert feed(client, data)["assessment_status"] == "ready"
