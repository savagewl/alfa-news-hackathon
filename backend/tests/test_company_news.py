import asyncio
import json
import time
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.config import Settings
from app.db.session import get_session
from app.main import create_app
from app.models.analytics import BankAnalyticsSelection
from app.models.company import Company
from app.schemas.company_news import InterpretationSource
from app.services.company_news import (
    DeadlineClient,
    LLMUnavailable,
    company_context,
    completed_response,
    generate_interpretation,
    prepare_interpretation,
    require_llm,
    validate_result,
)
from app.services.llm.config import ServicesSettings
from app.services.llm.schemas import (
    BankPassage,
    CompanyContext,
    InsightDraft,
    InsightResult,
    NewsEvent,
    Verification,
)


def test_request_cache_key_does_not_depend_on_retrieval(monkeypatch):
    context = CompanyContext(company_id="company", profile_text="Профиль")
    news = NewsEvent(url="https://example.test/news", text="Событие", summary="Событие")
    article = SimpleNamespace(
        id=uuid4(), title="Событие", canonical_url=news.url, published_at=None
    )
    monkeypatch.setattr("app.services.company_news.company_context", lambda _: context)
    monkeypatch.setattr(
        "app.repositories.news_embeddings.fetch_news_event",
        AsyncMock(return_value=(news, [], "news-fingerprint")),
    )
    passage = BankPassage(document_id=str(uuid4()), chunk_id="p1", title="Банк", text="Текст")
    monkeypatch.setattr(
        "app.services.company_news.retrieve_bank_passages",
        AsyncMock(side_effect=[([], [], []), ([passage], [], [])]),
    )

    async def prepare_twice():
        settings = ServicesSettings(_env_file=None)
        company = SimpleNamespace(okved_name=None)
        first = await prepare_interpretation(AsyncMock(), company, article, settings)
        second = await prepare_interpretation(AsyncMock(), company, article, settings)
        assert first[-1] == second[-1]
        assert first[2] != second[2]

    asyncio.run(prepare_twice())


def approved_result(news, company_id="company"):
    return InsightResult(
        draft=InsightDraft(
            headline="Событие для компании",
            company_id=company_id,
            event_id=news.url,
            facts=["Ставка изменилась"],
            ai_interpretation="Возможно изменение стоимости кредита",
            risks=["Стоимость финансирования может измениться"],
            suggested_actions=["Проверить условия кредитного договора"],
            claims=[
                {
                    "id": "c1",
                    "text": "Ставка изменилась",
                    "kind": "fact",
                    "evidence_refs": [f"news:{news.url}"],
                }
            ],
        ),
        verification=Verification(status="APPROVED"),
        steps=[],
    )


@pytest.fixture
def client():
    app = create_app(Settings(_env_file=None, rss_enabled=False, analytics_enabled=False))
    session = AsyncMock()
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as http:
        yield http, session


def test_company_news_requires_auth_and_exposes_both_routes(client):
    http, session = client
    assert http.get("/companies/c/news").status_code == 401
    assert http.post(f"/companies/c/news/{uuid4()}/interpretation").status_code == 401
    assert (
        http.post(f"/companies/c/news/{uuid4()}/interpretation/stream").status_code
        == 401
    )
    assert (
        http.post(
            "/companies/c/news/assessments/stream",
            json={"news_ids": [str(uuid4())]},
        ).status_code
        == 401
    )
    paths = http.get("/openapi.json").json()["paths"]
    assert paths["/companies/{company_id}/news"]["get"]["security"]
    interpretation = paths["/companies/{company_id}/news/{news_id}/interpretation"]["post"]
    detail_stream = paths[
        "/companies/{company_id}/news/{news_id}/interpretation/stream"
    ]["post"]
    stream = paths["/companies/{company_id}/news/assessments/stream"]["post"]
    assert interpretation["security"]
    assert detail_stream["security"]
    assert stream["security"]
    assert {"200", "202", "429", "502", "503", "504"} <= interpretation["responses"].keys()
    session.execute.assert_not_awaited()


def test_unknown_company_does_not_read_news_or_start_llm(client):
    http, session = client
    http.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=uuid4())
    session.scalar.return_value = None
    assert http.get("/companies/unknown/news").status_code == 404
    assert http.post(f"/companies/unknown/news/{uuid4()}/interpretation").status_code == 404
    # Only the shared bank selection is read; no news is loaded for an unknown company.
    assert session.get.await_count == 2
    for awaited in session.get.await_args_list:
        assert awaited.args == (BankAnalyticsSelection, 1)


def test_context_preserves_money_and_missing_information():
    values = dict.fromkeys(c.name for c in Company.__table__.columns)
    values.update(
        id="001",
        name=None,
        okved_code="01.01",
        okved_name="Растениеводство",
        annual_revenue=Decimal("12345678901234567890.12345"),
        has_credit=True,
        has_fts_data=None,
        updated_at=datetime.now(UTC),
    )
    context = company_context(SimpleNamespace(**values))
    assert "12345678901234567890.12345" in context.profile_text
    assert "Название: неизвестно" in context.profile_text
    assert context.products == ["loa"]
    assert "ВЭД" not in context.evidence_labels
    assert "Размер кредита" not in context.evidence_labels


def test_missing_provider_is_not_silently_replaced_with_demo():
    settings = ServicesSettings(_env_file=None)
    with pytest.raises(LLMUnavailable):
        require_llm(settings)
    require_llm(settings.model_copy(update={"llm_mock": True}))


def test_generation_uses_openai_compatible_http_and_existing_verifier(monkeypatch):
    news = NewsEvent(
        url="https://example.test/news", text="Ставка изменилась", summary="Ставка изменилась"
    )
    context = CompanyContext(company_id="company", profile_text="Компания company")
    calls = []

    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        if "проверяющий аналитик" in body["messages"][0]["content"]:
            content = {
                "status": "APPROVED",
                "remarks": [],
                "missing_evidence": [],
                "retry_queries": [],
            }
        else:
            content = approved_result(news).draft.model_dump(mode="json")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
        )

    def client(**kwargs):
        return httpx.Client(
            base_url=kwargs["base_url"],
            headers=kwargs["headers"],
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr("app.services.company_news.DeadlineClient", client)
    settings = ServicesSettings(
        _env_file=None, llm_api_base="https://llm.test/v1", llm_api_key="test-key"
    )
    result = generate_interpretation(context, news, [], settings)
    assert result.verification.status == "APPROVED"
    assert len(calls) == 2
    assert all(r.headers["authorization"] == "Bearer test-key" for r in calls)
    assert all(r.url.path == "/v1/chat/completions" for r in calls)


def test_expired_deadline_does_not_make_another_model_request():
    with DeadlineClient(
        deadline=time.monotonic() - 1,
        per_call_timeout=30,
        base_url="https://example.test",
        transport=httpx.MockTransport(
            lambda request: pytest.fail("Deadline must be checked first")
        ),
    ) as client:
        with pytest.raises(TimeoutError):
            client.post("/chat/completions", json={})


def test_unverified_bank_position_is_dropped_but_supported_interpretation_is_published():
    news = NewsEvent(url="https://example.test/news", text="Событие", summary="Событие")
    context = CompanyContext(company_id="company", profile_text="Профиль")
    result = approved_result(news)
    result.draft.bank_position = "Банк рекомендует кредит"
    result.draft.claims[0].evidence_refs.append("bank:invented#1")
    sources = [
        InterpretationSource(evidence_ref=f"news:{news.url}", title="Событие", source_url=news.url)
    ]
    checked = validate_result(result, context, news, sources)
    assert checked.verification.status == "APPROVED"
    assert checked.draft.bank_position is None
    assert checked.draft.claims[0].evidence_refs == [f"news:{news.url}"]
    assert any("Позиция банка не сформирована" in item for item in checked.draft.limitations)
    response = completed_response(
        uuid4(), "company", uuid4(), checked, sources, [], ServicesSettings(_env_file=None)
    )
    assert response.status == "ready"
    assert response.interpretation is not None


def test_interpretation_without_any_supported_claim_is_insufficient():
    news = NewsEvent(url="https://example.test/news", text="Событие", summary="Событие")
    context = CompanyContext(company_id="company", profile_text="Профиль")
    result = approved_result(news)
    result.draft.claims[0].evidence_refs = ["news:https://invented.example/"]
    sources = [
        InterpretationSource(evidence_ref=f"news:{news.url}", title="Событие", source_url=news.url)
    ]
    checked = validate_result(result, context, news, sources)
    assert checked.verification.status == "INSUFFICIENT"
    response = completed_response(
        uuid4(), "company", uuid4(), checked, sources, [], ServicesSettings(_env_file=None)
    )
    assert response.status == "insufficient"
    assert response.interpretation is None


def test_okved_keywords_skip_generic_words_that_match_any_production_news():
    from app.repositories.company_news import okved_keywords

    words = okved_keywords("Производство строительных металлических конструкций")
    assert "производство" not in words
    assert words == ["конструкций", "металлических", "строительных"]
    assert okved_keywords("Торговля оптовая компьютерами") == ["компьютерами"]
    assert okved_keywords(None) == []


def test_general_business_rule_does_not_make_news_relevant():
    from app.repositories.company_news import GENERAL_RULE, _has_specific_match

    rules = [("Совпадение с описанием деятельности компании", None, 0.5), (GENERAL_RULE, None, 0.1)]
    assert not _has_specific_match(rules, [False, True])
    assert _has_specific_match(rules, [True, False])
