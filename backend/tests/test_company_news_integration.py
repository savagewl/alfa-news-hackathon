import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.dependencies import get_current_user
from app.core.config import Settings
from app.main import create_app
from app.models.analytics import AnalyticsChunk, AnalyticsDocument
from app.models.company import Company, CompanyMembership
from app.models.interpretation import NewsInterpretation
from app.models.news import NewsArticle, NewsSource
from app.models.user import User
from app.services.llm.config import ServicesSettings, get_services_settings
from app.services.llm.schemas import AssessmentDraft
from tests.test_company_news import approved_result


@pytest.fixture
def settings():
    name = os.environ.get("COMPANY_NEWS_TEST_POSTGRES_DB")
    if not name:
        pytest.skip("Set COMPANY_NEWS_TEST_POSTGRES_DB to a dedicated migrated database")
    return Settings(postgres_db=name, rss_enabled=False, analytics_enabled=False)


@asynccontextmanager
async def sessions(settings):
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


def seed(settings):
    user_id, first_id, second_id = uuid4(), uuid4(), uuid4()
    company_id, source = uuid4().hex, "test-" + uuid4().hex

    async def insert_rows():
        async with sessions(settings) as factory, factory.begin() as session:
            session.add(User(user_id=user_id, email=f"{user_id}@example.com", password_hash="test"))
            session.add(
                NewsSource(
                    code=source, name="Источник", site_url="https://example.test", kind="media"
                )
            )
            session.add(
                Company(
                    id=company_id,
                    name="Компания",
                    okved_name="Производство компьютеров",
                    region="Татарстан",
                    has_credit=True,
                    is_placeholder=False,
                )
            )
            await session.flush()
            session.add(CompanyMembership(company_id=company_id, user_id=user_id))
            for identifier, title, topics, summary in (
                (
                    first_id,
                    "Производство компьютеров в Татарстане",
                    ["business"],
                    "Изменилось производство компьютеров",
                ),
                (second_id, "Футбольная команда победила", ["other"], None),
            ):
                session.add(
                    NewsArticle(
                        id=identifier,
                        source_code=source,
                        title=title,
                        summary=summary,
                        canonical_url=f"https://example.test/{identifier}",
                        original_url=f"https://example.test/{identifier}",
                        authors=[],
                        source_categories=[],
                        topics=topics,
                        content_hash=str(identifier),
                        normalizer_version=1,
                    )
                )

    asyncio.run(insert_rows())
    return SimpleNamespace(
        user_id=user_id,
        company_id=company_id,
        news_id=first_id,
        other_news_id=second_id,
        source=source,
    )


def api(settings, data, ai=None):
    app = create_app(settings)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=data.user_id)
    app.dependency_overrides[get_services_settings] = lambda: (
        ai or ServicesSettings(_env_file=None, llm_api_base="https://llm.test/v1")
    )
    return TestClient(app)


def route(data):
    return f"/companies/{data.company_id}/news/{data.news_id}/interpretation"


def test_fast_feed_rules_pagination_and_no_llm(settings, monkeypatch):
    data = seed(settings)
    monkeypatch.setattr(
        "app.api.company_news.generate_interpretation",
        lambda *a: pytest.fail("Feed must not call LLM"),
    )
    with api(settings, data) as client:
        url = f"/companies/{data.company_id}/news"
        result = client.get(url, params={"source": data.source})
        assert result.status_code == 200, result.text
        page = result.json()
        assert page["total"] == 1
        assert page["items"][0]["id"] == str(data.news_id)
        assert page["items"][0]["relevance_score"] > 0.5
        assert page["items"][0]["summary_origin"] == "rss"
        all_news = client.get(url, params={"source": data.source, "relevant_only": False}).json()
        assert all_news["total"] == 2
        assert all_news["items"][1]["summary_origin"] == "title"
        assert client.get(url, params={"source": data.source, "offset": 1}).json()["items"] == []
        assert client.get(url + "?limit=101").status_code == 422
        assert client.get(url + "?published_from=2026-01-01").status_code == 422


def test_generation_cache_and_input_invalidation(settings, monkeypatch):
    data = seed(settings)
    calls = []

    def generate(context, news, passages, ai):
        calls.append(context.profile_text)
        assert not passages
        return approved_result(news)

    monkeypatch.setattr("app.api.company_news.generate_interpretation", generate)
    with api(settings, data) as client:
        response = client.post(route(data))
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] == "ready" and result["reused"] is False
        assert result["interpretation"]["company_id"] == data.company_id
        assert result["interpretation"]["bank_position"] is None
        assert result["warnings"]
        assert client.post(route(data)).json()["reused"] is True
        assert len(calls) == 1

        async def change():
            async with sessions(settings) as factory, factory.begin() as session:
                await session.execute(
                    update(Company).where(Company.id == data.company_id).values(city="Казань")
                )

        asyncio.run(change())
        assert client.post(route(data)).json()["reused"] is False
        assert len(calls) == 2


def test_company_membership_checked_before_cache_or_news_lookup(settings, monkeypatch):
    data = seed(settings)
    monkeypatch.setattr(
        "app.api.company_news.generate_interpretation", lambda c, n, *a: approved_result(n)
    )
    with api(settings, data) as client:
        assert client.post(route(data)).status_code == 200
        client.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=uuid4())
        assert client.post(route(data)).status_code == 404
        assert client.get(f"/companies/{data.company_id}/news").status_code == 404


def test_missing_llm_and_upstream_failures_are_explicit_and_retryable(settings, monkeypatch):
    data = seed(settings)
    with api(settings, data, ServicesSettings(_env_file=None)) as client:
        result = client.post(route(data))
        assert result.status_code == 503
        assert "LLM is not configured" in result.json()["detail"]

    def fail(*args):
        raise RuntimeError("private provider details")

    monkeypatch.setattr("app.api.company_news.generate_interpretation", fail)
    with api(settings, data) as client:
        response = client.post(route(data))
        assert response.status_code == 502 and "private" not in response.text
        monkeypatch.setattr(
            "app.api.company_news.generate_interpretation", lambda c, n, *a: approved_result(n)
        )
        assert client.post(route(data)).json()["status"] == "ready"


def test_simultaneous_requests_generate_once_and_leave_feed_responsive(settings, monkeypatch):
    data = seed(settings)
    started, finish = threading.Event(), threading.Event()
    calls = []

    def generate(context, news, *args):
        calls.append(news.url)
        started.set()
        assert finish.wait(5)
        return approved_result(news)

    monkeypatch.setattr("app.api.company_news.generate_interpretation", generate)
    with api(settings, data) as client, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, route(data))
        assert started.wait(5)
        try:
            second = client.post(route(data))
            assert second.status_code == 202 and second.headers["retry-after"] == "2"
            assert client.get(f"/companies/{data.company_id}/news").status_code == 200
        finally:
            finish.set()
        assert first.result(timeout=10).status_code == 200
        assert len(calls) == 1


def test_unapproved_draft_is_not_published(settings, monkeypatch):
    data = seed(settings)

    def generate(context, news, *args):
        result = approved_result(news)
        result.verification.status = "REVISE"
        return result

    monkeypatch.setattr("app.api.company_news.generate_interpretation", generate)
    with api(settings, data) as client:
        result = client.post(route(data)).json()
        assert result["status"] == "needs_revision" and result["interpretation"] is None


def test_bank_retrieval_only_uses_selected_dated_documents(settings, monkeypatch):
    data = seed(settings)
    past, future, unknown, unselected = [uuid4() for _ in range(4)]

    async def add_documents():
        async with sessions(settings) as factory, factory.begin() as session:
            for identifier, published in (
                (past, datetime.now(UTC).date() - timedelta(days=1)),
                (future, datetime.now(UTC).date() + timedelta(days=1)),
                (unknown, None),
                (unselected, datetime.now(UTC).date()),
            ):
                session.add(
                    AnalyticsDocument(
                        id=identifier,
                        sha256=str(identifier),
                        filename="test.pdf",
                        title="Производство компьютеров",
                        publisher="Альфа-Банк",
                        published_at=published,
                        date_origin="provided",
                        uploaded_by=data.user_id,
                        content=b"%PDF-test",
                        status="parsed",
                        page_count=7,
                        chunk_count=1,
                    )
                )
                await session.flush()
                session.add(
                    AnalyticsChunk(
                        document_id=identifier,
                        chunk_id="p7-1",
                        page_number=7,
                        section="Прогноз",
                        text="Прогноз: производство компьютеров может увеличиться",
                        search_text="Прогноз производство компьютеров",
                        bbox=[],
                    )
                )

    asyncio.run(add_documents())

    def generate(context, news, passages, ai):
        assert len(passages) == 1 and passages[0].document_id == str(past)
        result = approved_result(news)
        result.draft.bank_position = "Банк допускает увеличение производства [B1]"
        result.draft.claims.append(
            result.draft.claims[0].model_copy(
                update={
                    "id": "bank",
                    "kind": "bank_position",
                    "text": result.draft.bank_position,
                    "evidence_refs": [f"bank:{past}#p7-1"],
                }
            )
        )
        return result

    monkeypatch.setattr("app.api.company_news.generate_interpretation", generate)
    ai = ServicesSettings(
        _env_file=None,
        llm_api_base="https://llm.test/v1",
        bank_document_ids=[past, future, unknown],
    )
    with api(settings, data, ai) as client:
        response = client.post(route(data))
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["status"] == "ready" and payload["interpretation"]["bank_position"]
        bank_sources = [s for s in payload["sources"] if s["evidence_ref"].startswith("bank:")]
        assert len(bank_sources) == 1
        assert bank_sources[0]["source_url"] == f"/analytics/documents/{past}/file#page=7"
        card = client.get(
            f"/companies/{data.company_id}/news",
            params={"source": data.source},
        ).json()["items"][0]
        assert card["bank_analytics"] == payload["interpretation"]["bank_position"]
        assert card["bank_sources"] == bank_sources


def test_deadline_and_busy_limit_do_not_cache_failed_generation(settings, monkeypatch):
    data = seed(settings)

    def slow(context, news, *args):
        time.sleep(0.15)
        return approved_result(news)

    monkeypatch.setattr("app.api.company_news.generate_interpretation", slow)
    ai = ServicesSettings(_env_file=None, llm_api_base="https://llm.test/v1").model_copy(
        update={"interpretation_timeout_seconds": 0.05}
    )
    with api(settings, data, ai) as client:
        assert client.post(route(data)).status_code == 504
    with api(settings, data) as client:
        client.app.state.interpretation_slots = asyncio.Semaphore(0)
        assert client.post(route(data)).status_code == 429
        client.app.state.interpretation_slots = asyncio.Semaphore(1)
        monkeypatch.setattr(
            "app.api.company_news.generate_interpretation", lambda c, n, *a: approved_result(n)
        )
        assert client.post(route(data)).json()["status"] == "ready"


def test_expired_cache_and_missing_news(settings, monkeypatch):
    data = seed(settings)
    calls = []

    def generate(context, news, *args):
        calls.append(news.url)
        return approved_result(news)

    monkeypatch.setattr("app.api.company_news.generate_interpretation", generate)
    with api(settings, data) as client:
        assert client.post(route(data)).json()["status"] == "ready"

        async def expire():
            async with sessions(settings) as factory, factory.begin() as session:
                await session.execute(
                    update(NewsInterpretation)
                    .where(NewsInterpretation.company_id == data.company_id)
                    .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
                )

        asyncio.run(expire())
        assert client.post(route(data)).json()["reused"] is False
        assert len(calls) == 2
        assert (
            client.post(f"/companies/{data.company_id}/news/{uuid4()}/interpretation").status_code
            == 404
        )


def test_chat_requires_current_insight_and_enforces_owner(settings, monkeypatch):
    data = seed(settings)
    monkeypatch.setattr(
        "app.api.company_news.generate_interpretation",
        lambda context, news, passages, ai: approved_result(news),
    )
    with api(settings, data) as client:
        base = f"/companies/{data.company_id}/news/{data.news_id}/chat/threads"
        thread_response = client.post(base)
        assert thread_response.status_code == 201, thread_response.text
        thread_id = thread_response.json()["id"]
        messages = f"{base}/{thread_id}/messages"
        missing = client.post(
            messages,
            json={"question": "Что это значит?", "client_message_id": str(uuid4())},
        )
        assert missing.status_code == 409
        assert client.get(messages).json()["messages"] == []

        assert client.post(route(data)).status_code == 200
        client.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
            user_id=uuid4()
        )
        assert client.get(messages).status_code == 404
        assert (
            client.post(
                messages,
                json={"question": "Чужой чат", "client_message_id": str(uuid4())},
            ).status_code
            == 404
        )


def test_chat_message_idempotency_and_history_cursor(settings, monkeypatch):
    data = seed(settings)
    monkeypatch.setattr(
        "app.api.company_news.generate_interpretation",
        lambda context, news, passages, ai: approved_result(news),
    )
    calls = []

    def answer(context, news, insight, passages, sources, question, history, ai):
        calls.append((question, history))
        return (
            f"Ответ: {question}",
            "ai_inference",
            [],
            ["Недостаточно данных: тестовый ответ"],
        )

    monkeypatch.setattr("app.api.chat.generate_chat_answer", answer)
    with api(settings, data) as client:
        assert client.post(route(data)).status_code == 200
        # A completed interpretation must support chat even when search is unavailable.
        monkeypatch.setattr(
            "app.services.company_news.retrieve_bank_passages",
            lambda *a, **kw: pytest.fail("Cached interpretation must not repeat retrieval"),
        )
        # Previously saved rows used retrieval-dependent request keys.
        async def legacy_key():
            async with sessions(settings) as factory, factory.begin() as session:
                await session.execute(
                    update(NewsInterpretation)
                    .where(NewsInterpretation.company_id == data.company_id)
                    .values(request_fingerprint=f"legacy-{uuid4()}")
                )

        asyncio.run(legacy_key())
        assert client.post(route(data)).json()["reused"] is True
        base = f"/companies/{data.company_id}/news/{data.news_id}/chat/threads"
        thread_id = client.post(base).json()["id"]
        messages = f"{base}/{thread_id}/messages"

        first_id = uuid4()
        first_body = {"question": "Первый вопрос", "client_message_id": str(first_id)}
        first = client.post(messages, json=first_body)
        assert first.status_code == 200, first.text
        repeated = client.post(messages, json=first_body)
        assert repeated.status_code == 200
        assert repeated.json()["reused"] is True
        assert repeated.json()["assistant_message"]["id"] == first.json()["assistant_message"]["id"]
        assert len(calls) == 1

        second = client.post(
            messages,
            json={"question": "Второй вопрос", "client_message_id": str(uuid4())},
        )
        assert second.status_code == 200, second.text
        assert "Первый вопрос" in calls[-1][1]

        latest = client.get(messages, params={"limit": 2}).json()
        assert latest["has_more"] is True and latest["next_before"]
        older = client.get(
            messages, params={"limit": 2, "before": latest["next_before"]}
        ).json()
        assert older["has_more"] is False
        assert len(older["messages"]) == 2
        assert {item["id"] for item in older["messages"]}.isdisjoint(
            item["id"] for item in latest["messages"]
        )


def test_assessment_stream_short_assessments_cache_and_item_failure(settings, monkeypatch):
    data = seed(settings)
    calls = []

    def generate(context, news, passages, ai, **kwargs):
        calls.append(news.url)
        if str(data.other_news_id) in news.url:
            raise RuntimeError("test failure")
        time.sleep(0.05)
        return AssessmentDraft(
            brief_impact="Персонализированная сводка.",
            risks="Риск для компании.", opportunities="Прямых возможностей нет."
        )

    monkeypatch.setattr("app.services.assessments.generate_assessment", generate)

    def events(response):
        parsed = []
        for block in response.text.strip().split("\n\n"):
            lines = block.splitlines()
            parsed.append((lines[0].removeprefix("event: "), json.loads(lines[1][6:])))
        return parsed

    url = f"/companies/{data.company_id}/news/assessments/stream"
    with api(settings, data) as client:
        response = client.post(
            url,
            json={"news_ids": [str(data.news_id), str(data.other_news_id)]},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        stream = events(response)
        assert [item["status"] for event, item in stream[:2]] == ["queued", "queued"]
        terminal = {
            item["news_id"]: item
            for event, item in stream
            if event == "assessment" and item["status"] in {"ready", "failed"}
        }
        ready = terminal[str(data.news_id)]
        assert ready["status"] == "ready" and ready["reused"] is False
        assert ready["risks"] == ["Риск для компании."]
        assert ready["brief_impact"] == "Персонализированная сводка."
        # «возможностей нет» — отсутствие данных: null, фронт блок не выводит
        assert ready["opportunities"] is None
        assert terminal[str(data.other_news_id)]["status"] == "failed"
        assert terminal[str(data.other_news_id)]["retryable"] is True
        assert stream[-1][0] == "done"
        assert stream[-1][1]["counts"] == {"failed": 1, "ready": 1}

        before = len(calls)
        cached = events(client.post(url, json={"news_ids": [str(data.news_id)]}))
        reused = next(item for event, item in cached if item["status"] == "ready")
        assert reused["reused"] is True and reused["risks"] == ["Риск для компании."]
        assert reused["brief_impact"] == ready["brief_impact"]
        assert len(calls) == before


def test_interpretation_stream_progress_cache_and_error(settings, monkeypatch):
    data = seed(settings)
    ai = ServicesSettings(_env_file=None, llm_mock=True)

    def events(response):
        parsed = []
        for block in response.text.strip().split("\n\n"):
            lines = block.splitlines()
            if lines[0].startswith(":"):
                continue
            parsed.append((lines[0].removeprefix("event: "), json.loads(lines[1][6:])))
        return parsed

    url = f"/companies/{data.company_id}/news/{data.news_id}/interpretation/stream"
    with api(settings, data, ai) as client:
        first = events(client.post(url))
        assert first[0] == ("progress", {"stage": "preparing"})
        draft_index = next(index for index, (event, _) in enumerate(first) if event == "draft")
        result_index = next(index for index, (event, _) in enumerate(first) if event == "result")
        assert draft_index < result_index
        assert first[draft_index][1]["verified"] is False
        assert first[draft_index][1]["revision_version"] == 0
        assert first[result_index][1]["status"] == "insufficient"
        assert first[result_index][1]["interpretation"] is None
        assert first[-1][0] == "done"

        cached = events(client.post(url))
        assert [event for event, _ in cached] == ["progress", "result", "done"]
        assert cached[1][1]["reused"] is True

        monkeypatch.setattr(
            "app.api.company_news.generate_interpretation",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private")),
        )
        error_url = (
            f"/companies/{data.company_id}/news/{data.other_news_id}/interpretation/stream"
        )
        failed = events(client.post(error_url))
        assert failed[-2][0] == "error" and "private" not in failed[-2][1]["detail"]
        assert failed[-1][0] == "done"
        
def test_generic_okved_words_do_not_make_unrelated_news_relevant(settings):
    from sqlalchemy import select

    from app.repositories.company_news import relevance_rules

    data = seed(settings)
    titles = {
        "Аренда и лизинг легковых автомобилей вырос на треть": ["business"],
        "Минобороны сообщило об уничтожении беспилотников и средств ПВО": ["other"],
        "Ставки на перевозку российской нефти выросли до рекордного уровня": ["other"],
        "Банк России сохранил ключевую ставку на уровне 14%": ["finance"],
    }
    ids = {}

    async def scenario():
        async with sessions(settings) as factory, factory.begin() as session:
            company = await session.get(Company, data.company_id)
            company.okved_name = "Аренда и лизинг легковых автомобилей и легких средств"
            company.has_leasing = True
            company.region = None
            for title, topics in titles.items():
                identifier = uuid4()
                ids[title] = identifier
                session.add(
                    NewsArticle(
                        id=identifier,
                        source_code=data.source,
                        title=title,
                        summary=None,
                        canonical_url=f"https://example.test/{identifier}",
                        original_url=f"https://example.test/{identifier}",
                        authors=[],
                        source_categories=[],
                        topics=topics,
                        content_hash=str(identifier),
                        normalizer_version=1,
                    )
                )
            await session.flush()
            rules = relevance_rules(company)
            rows = await session.execute(
                select(NewsArticle.id, *(condition for _, condition, _ in rules)).where(
                    NewsArticle.id.in_(ids.values())
                )
            )
            names = [name for name, _, _ in rules]
            return {
                row[0]: {n for n, flag in zip(names, row[1:], strict=True) if flag} for row in rows
            }

    matched = asyncio.run(scenario())
    activity = "Совпадение с описанием деятельности компании"
    finance = "Финансовая повестка при наличии банковских продуктов"
    leasing = matched[ids["Аренда и лизинг легковых автомобилей вырос на треть"]]
    assert activity in leasing
    drones = matched[ids["Минобороны сообщило об уничтожении беспилотников и средств ПВО"]]
    assert activity not in drones and finance not in drones
    freight = matched[ids["Ставки на перевозку российской нефти выросли до рекордного уровня"]]
    assert activity not in freight and finance not in freight
    key_rate = matched[ids["Банк России сохранил ключевую ставку на уровне 14%"]]
    assert finance in key_rate
