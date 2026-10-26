import asyncio
import json
from uuid import uuid4

import pytest

from app.repositories import company_news as feed_repository
from app.schemas.company_news import InterpretationSource
from app.services import assessments
from app.services.llm.config import ServicesSettings
from app.services.llm.provider import MockProvider
from app.services.llm.schemas import (
    AssessmentDraft,
    BankPassage,
    CompanyContext,
    CompanyFields,
    CompanyProfile,
    CompanyProfileText,
    NewsAnalysis,
    NewsEvent,
    ProcessedNews,
)


def _passage(number: int) -> BankPassage:
    return BankPassage(
        document_id=str(uuid4()), chunk_id=f"p{number}", title=f"Отчёт {number}", text="текст"
    )


def _source(passage: BankPassage) -> InterpretationSource:
    return InterpretationSource(
        evidence_ref=f"bank:{passage.document_id}#{passage.chunk_id}",
        title=passage.title,
        source_url="/analytics/documents/x/file#page=1",
        page_number=1,
    )


def test_shorten_keeps_two_sentences_and_limits_length():
    text = "Раз. Два! Три? Четыре. Пять. Шесть."
    assert assessments.shorten(text) == "Раз. Два!"
    long = "слово " * 400
    result = assessments.shorten(long)
    assert len(result) <= assessments.MAX_BLOCK_CHARS + 1 and result.endswith("…")
    assert assessments.shorten("  Одно   предложение.  ") == "Одно предложение."


def test_finalize_keeps_bank_analytics_only_with_valid_references():
    passages = [_passage(1), _passage(2)]
    sources = [_source(passage) for passage in passages]
    draft = AssessmentDraft(
        brief_impact="Влияние на компанию.",
        risks="Риск.", opportunities="Возможность.", bank_analytics="Банк.", bank_refs=[2, 2, 9]
    )
    result = assessments.finalize(draft, passages, sources)
    assert result["brief_impact"] == "Влияние на компанию."
    assert result["bank_analytics"] == "Банк."
    assert [item["evidence_ref"] for item in result["bank_sources"]] == [sources[1].evidence_ref]
    assert [item["chunk_id"] for item in result["bank_passages"]] == ["p2"]

    for bad in (
        draft.model_copy(update={"bank_refs": []}),
        draft.model_copy(update={"bank_refs": [5]}),
        draft.model_copy(update={"bank_analytics": None}),
    ):
        result = assessments.finalize(bad, passages, sources)
        assert result["bank_analytics"] is None
        assert result["bank_sources"] == [] and result["bank_passages"] == []

    no_passages = assessments.finalize(draft, [], [])
    assert no_passages["bank_analytics"] is None


def test_blank_or_nothing_blocks_become_none_for_the_feed():
    draft = AssessmentDraft(risks="  ", opportunities="Прямых возможностей нет.")
    assert draft.risks is None and draft.opportunities is None
    view = assessments.assessment_view(draft.model_dump())
    assert view["risks"] is None and view["opportunities"] is None
    filled = AssessmentDraft(risks="Есть.", opportunities=None)
    assert filled.risks == "Есть." and filled.opportunities is None
    payload = assessments.finalize(filled, [], [])
    assert payload["risks"] == "Есть." and payload["opportunities"] is None


def test_prompt_contains_profile_news_and_numbered_bank_passages():
    context = CompanyContext(company_id="U1", profile_text="Отрасль: ИТ")
    news = NewsEvent(title="Ставка", summary="ЦБ снизил ставку", text="Полный текст " * 500)
    prompt = assessments.build_messages(context, news, [_passage(1), _passage(2)])
    body = prompt[-1]["content"]
    assert prompt[0]["role"] == "system" and "bank_refs" in prompt[0]["content"]
    assert "Отрасль: ИТ" in body and "ЦБ снизил ставку" in body
    assert "[B1] Отчёт 1" in body and "[B2] Отчёт 2" in body
    assert len(body) < assessments.NEWS_TEXT_CHARS + 3000
    empty = assessments.build_messages(context, news, [])[-1]["content"]
    assert "релевантных материалов нет" in empty


def test_generate_assessment_uses_provider_and_repairs_invalid_json():
    answers = iter(
        [
            "не json",
            json.dumps({"risks": "Риск.", "opportunities": "Шанс.", "bank_analytics": None}),
        ]
    )
    calls = []

    def handler(purpose, messages, json_mode):
        calls.append(purpose)
        return next(answers)

    settings = ServicesSettings(_env_file=None, llm_api_base="https://llm.test/v1")
    draft = assessments.generate_assessment(
        CompanyContext(company_id="U1", profile_text="p"),
        NewsEvent(summary="s", text="t"),
        [],
        settings,
        llm=MockProvider(handler),
    )
    assert draft.risks == "Риск." and calls == ["news_assessment"] * 2


def test_generate_assessment_in_mock_mode_returns_demo_text():
    settings = ServicesSettings(_env_file=None, llm_mock=True)
    draft = assessments.generate_assessment(
        CompanyContext(company_id="U1", profile_text="p"),
        NewsEvent(summary="s", text="t"),
        [],
        settings,
    )
    assert "Демо-режим" in draft.risks and draft.bank_analytics is None


def test_assessment_view_wraps_blocks_for_the_feed():
    view = assessments.assessment_view(
        {
            "brief_impact": "Влияние на компанию.",
            "risks": "Риск.",
            "opportunities": "Шанс.",
            "bank_analytics": "Банк.",
            "bank_sources": [{"evidence_ref": "bank:x#1"}],
        }
    )
    assert view["assessment_status"] == "ready"
    assert view["brief_impact"] == "Влияние на компанию."
    assert view["risks"] == ["Риск."] and view["opportunities"] == ["Шанс."]
    assert view["bank_analytics"] == "Банк." and view["bank_sources"]


def _scored_feed():
    profile = CompanyProfile(
        text=CompanyProfileText(synthetic_u="U1", main_text="профиль компании", aspects={}),
        vectors={"main": [1.0, 0.0]},
        embedding_model_version="bge-m3:2d",
    )
    fields = CompanyFields(synthetic_u="U1")
    news, semantic = {}, {}
    settings = ServicesSettings(_env_file=None, rerank_top_k=2)
    for index, vector in enumerate(([1.0, 0.0], [0.6, 0.8], [0.0, 1.0])):
        article_id = uuid4()
        news[article_id] = ProcessedNews(
            url=f"https://example.test/{index}",
            text="текст",
            summary=f"суть {index}",
            markup=NewsAnalysis(summary=f"суть {index}"),
            vector=vector,
            embedding_model_version="bge-m3:2d",
        )
        semantic[article_id] = feed_repository.score_relevance(
            profile, fields, news[article_id], settings=settings, news_id=str(article_id)
        )
    return profile, fields, news, semantic, settings


class StubReranker:
    def __init__(self, enabled=True, scores=None):
        self.enabled = enabled
        self.scores = scores
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        return self.scores(documents) if self.scores else [None] * len(documents)


def test_feed_reranks_only_top_candidates(monkeypatch):
    profile, fields, news, semantic, settings = _scored_feed()
    before = {key: value.score for key, value in semantic.items()}
    stub = StubReranker(scores=lambda docs: [0.9] * len(docs))
    monkeypatch.setattr(feed_repository, "get_reranker", lambda _settings: stub)
    asyncio.run(feed_repository._apply_reranker(profile, fields, semantic, news, settings))
    assert len(stub.calls) == 1 and stub.calls[0][0] == "профиль компании"
    assert stub.calls[0][1] == ["суть 0", "суть 1"]  # top-2 by current score
    ids = list(news)
    assert semantic[ids[0]].score == pytest.approx(settings.relevance_cosine_weight * 0.9)
    assert semantic[ids[1]].score == pytest.approx(settings.relevance_cosine_weight * 0.9)
    assert semantic[ids[2]].score == before[ids[2]]


def test_feed_keeps_cosine_scores_when_reranker_fails_or_is_disabled(monkeypatch):
    profile, fields, news, semantic, settings = _scored_feed()
    before = {key: value.score for key, value in semantic.items()}
    for stub in (StubReranker(enabled=False), StubReranker()):
        monkeypatch.setattr(feed_repository, "get_reranker", lambda _settings, stub=stub: stub)
        asyncio.run(feed_repository._apply_reranker(profile, fields, semantic, news, settings))
        assert {key: value.score for key, value in semantic.items()} == before


def test_feed_texts_are_short():
    from app.repositories.interpretations import _brief, _brief_list

    assert _brief("Раз. Два. Три.", sentences=2, chars=280) == "Раз. Два."
    assert _brief("  ", sentences=2, chars=280) is None
    items = _brief_list(["Первый риск. Подробности.", "Второй.", "Третий."])
    assert items == ["Первый риск.", "Второй."]


def test_empty_assessment_statements_become_none():
    from app.services.llm.text import is_empty_assessment

    for text in (
        None,
        "  ",
        "null",
        "Рисков нет.",
        "Прямых рисков для компании нет, так как её деятельность связана с лизингом.",
        "Новость не несет значимых возможностей для бизнеса в сфере аренды.",
        "Возможностей для компании нет.",
        "Не выявлено",
        "Рисков и возможностей не выявлено",
        "Значимых рисков для компании не выявлено.",
    ):
        assert is_empty_assessment(text), text
    for text in (
        "Возможно замедление платежей со стороны заказчиков.",
        "Рост ставок может увеличить стоимость лизинга и снизить спрос.",
        "Нет гарантии, что платежи останутся прежними, поэтому риск высок.",
    ):
        assert not is_empty_assessment(text), text
