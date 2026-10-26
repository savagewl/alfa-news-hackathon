import json

from app.services.llm.config import ServicesSettings
from app.services.llm.insight import _normalize_evidence_refs, answer_question, build_insight
from app.services.llm.provider import LLMResponse, MockProvider
from app.services.llm.schemas import (
    BankPassage,
    CompanyContext,
    InsightClaim,
    InsightDraft,
    NewsEvent,
)


def _env_free_settings(**overrides) -> ServicesSettings:
    return ServicesSettings(_env_file=None, **overrides)


def _context() -> CompanyContext:
    return CompanyContext(
        company_id="U_87",
        profile_text="Компания U_87; Отрасль (ОКВЭД 25.11): производство металлоконструкций",
        okved_code="25.11",
        client_type="CLIENT",
        products=["loa", "gar"],
        region="Курганская область",
        counterparties_text="ОАНО Стальпро (входящие)",
        relevance_reasons=["Совпадение класса ОКВЭД 25 (+0.35)"],
    )


def _news() -> NewsEvent:
    return NewsEvent(
        url="https://example.ru/rate-14",
        title="Ставка сохранена на 14%",
        published_at="2026-09-11",
        text=(
            "Совет директоров Банка России сохранил ключевую ставку на уровне 14%. "
            "Риторика регулятора стала жёстче."
        ),
        summary="Банк России сохранил ключевую ставку 14%: пауза в цикле смягчения.",
    )


def _passages() -> list[BankPassage]:
    return [
        BankPassage(
            document_id="rate_report_14092026",
            chunk_id="3",
            title="Ключевая ставка ЦБ РФ: ставка без изменения на 14,0%",
            text="Совет директоров сохранил ставку на 14,0%, риторика стала жёстче.",
            published_at="2026-09-14",
        )
    ]


def _draft_json() -> str:
    return json.dumps(
        {
            "headline": "Ставка 14%: удешевления кредитов ждать не стоит",
            "facts": ["Ставка сохранена на уровне 14%"],
            "bank_position": "Регулятор сохраняет ставку, риторика жёстче [B1]",
            "ai_interpretation": (
                "Для компании U_87 с кредитными продуктами период быстрого удешевления "
                "заемных средств приостановлен"
            ),
            "risks": ["Финансовая нагрузка не снизится"],
            "opportunities": [],
            "suggested_actions": ["Пересмотреть график рефинансирования"],
            "limitations": ["Нет данных о текущей долговой нагрузке компании"],
            "claims": [
                {
                    "id": "c1",
                    "text": "Ставка сохранена на 14%",
                    "kind": "bank_position",
                    "evidence_refs": ["bank:rate_report_14092026#3"],
                }
            ],
        },
        ensure_ascii=False,
    )


def test_build_insight_approved(mock_provider):
    result = build_insight(_context(), _news(), _passages(), llm=mock_provider)
    assert result.verification.status == "APPROVED"
    assert result.draft.company_id == "U_87"
    assert result.draft.event_id == "https://example.ru/rate-14"
    assert [step.step for step in result.steps] == ["analyst", "verifier"]


def test_build_insight_revise_loop():
    draft = _draft_json()
    events: list[str] = []
    progress: list[tuple[str, dict]] = []

    def handler(purpose, messages, json_mode):
        if purpose == "insight_analyst":
            return draft
        if purpose == "insight_verifier":
            assert any(event == "draft" for event, _ in progress)
            events.append("verify")
            return json.dumps(
                {
                    "status": "REVISE",
                    "remarks": ["Добавь limitations"],
                    "missing_evidence": [],
                    "retry_queries": [],
                },
                ensure_ascii=False,
            )
        if purpose == "insight_revise":
            events.append("revise")
            return draft
        raise AssertionError(f"unexpected purpose {purpose}")

    result = build_insight(
        _context(),
        _news(),
        _passages(),
        llm=MockProvider(handler=handler),
        settings=_env_free_settings(insight_max_cycles=1),
        progress_callback=lambda event, data: progress.append((event, data)),
    )
    assert [step.step for step in result.steps] == ["analyst", "verifier", "revise", "verifier"]
    assert result.verification.status == "REVISE"
    assert events == ["verify", "revise", "verify"]
    snapshots = [data for event, data in progress if event == "draft"]
    assert [item["revision_version"] for item in snapshots] == [0, 1]
    resets = [data for event, data in progress if event == "reset"]
    assert [item["revision_version"] for item in resets] == [0, 1]
    assert all(item["verified"] is False for item in snapshots)
    assert [
        data["stage"] for event, data in progress if event == "progress"
    ] == ["generating", "verifying", "revising", "verifying"]


def test_build_insight_emits_streamed_field_deltas_before_draft():
    draft = _draft_json()

    class StreamingProvider:
        def complete_stream(self, messages, *, model, purpose, content_callback, **kwargs):
            assert purpose == "insight_analyst"
            for char in draft:
                content_callback(char)
            return LLMResponse(content=draft, model=model, usage_available=False)

        def complete(self, messages, *, model, purpose, **kwargs):
            assert purpose == "insight_verifier"
            return LLMResponse(
                content=json.dumps(
                    {
                        "status": "APPROVED",
                        "remarks": [],
                        "missing_evidence": [],
                        "retry_queries": [],
                    }
                ),
                model=model,
            )

    progress: list[tuple[str, dict]] = []
    build_insight(
        _context(),
        _news(),
        _passages(),
        llm=StreamingProvider(),
        settings=_env_free_settings(),
        progress_callback=lambda event, data: progress.append((event, data)),
    )
    reset_index = next(index for index, item in enumerate(progress) if item[0] == "reset")
    delta_indexes = [index for index, item in enumerate(progress) if item[0] == "delta"]
    draft_index = next(index for index, item in enumerate(progress) if item[0] == "draft")
    assert reset_index < min(delta_indexes) < draft_index
    assert {
        item["revision_version"] for event, item in progress if event in {"reset", "delta", "draft"}
    } == {0}
    interpretation = "".join(
        item["text"]
        for event, item in progress
        if event == "delta" and item["field"] == "ai_interpretation"
    )
    assert interpretation == json.loads(draft)["ai_interpretation"]
    assert not any(
        item.get("field") == "claims" for event, item in progress if event == "delta"
    )


def test_answer_question_uses_mock(mock_provider):
    draft = InsightDraft.model_validate_json(_draft_json())
    answer, step = answer_question(
        _context(), _news(), draft, "Стоит ли рефинансировать кредит?", llm=mock_provider
    )
    assert "Демо-режим" in answer
    assert step.step == "followup"


def test_answer_question_appends_related_news():
    draft = InsightDraft.model_validate_json(_draft_json())
    related = [
        NewsEvent(
            url="https://example.ru/provider-expansion",
            title="Поставщик расширяется",
            text="Компания Stalpro открывает подразделение и готовится к крупным закупкам.",
            summary="Поставщик расширяется и планирует крупные закупки.",
        )
    ]
    captured: dict[str, str] = {}

    def handler(purpose, messages, json_mode):
        captured[purpose] = messages[-1]["content"]
        return "Демо-ответ"

    answer, _ = answer_question(
        _context(),
        _news(),
        draft,
        "Как поступить с закупками?",
        llm=MockProvider(handler=handler),
        related_news=related,
    )
    followup_prompt = captured["followup"]
    assert "=== ДРУГИЕ РЕЛЕВАНТНЫЕ НОВОСТИ ===" in followup_prompt
    assert "provider-expansion" in followup_prompt
    assert followup_prompt.count("rate-14") == 1  # основная новость в блоке НОВОСТЬ, в related нет


def test_normalize_evidence_refs_fixes_and_drops():
    draft = InsightDraft(
        headline="x",
        ai_interpretation="x",
        claims=[
            InsightClaim(
                id="c1",
                text="t",
                kind="fact",
                evidence_refs=[
                    "company:Сорма собственности",
                    "news:https://example.ru/n",
                    "company:Выручка",
                ],
            ),
            InsightClaim(
                id="c2",
                text="t2",
                kind="fact",
                evidence_refs=["company:Что-то невнятное"],
            ),
        ],
    )
    fixed, note = _normalize_evidence_refs(draft)
    assert fixed.claims[0].evidence_refs == [
        "company:Форма собственности",
        "news:https://example.ru/n",
        "company:Выручка",
    ]
    assert fixed.claims[1].evidence_refs == ["company:Что-то невнятное"]  # не опустошаем claim
    assert "исправлено 1: Сорма собственности→Форма собственности" in note
    assert "удалено 1: Что-то невнятное" in note


def test_normalize_evidence_refs_accepts_dynamic_labels():
    draft = InsightDraft(
        headline="x",
        ai_interpretation="x",
        claims=[
            InsightClaim(
                id="c1",
                text="t",
                kind="fact",
                evidence_refs=["company:Отрасль (ОКВЭД 35.12)"],
            )
        ],
    )
    normalized, note = _normalize_evidence_refs(draft, ["Отрасль (ОКВЭД 35.12)"])
    assert normalized.claims[0].evidence_refs == ["company:Отрасль (ОКВЭД 35.12)"]
    assert note is None


def test_normalize_evidence_refs_keeps_clean_draft():
    draft = InsightDraft.model_validate_json(_draft_json())
    normalized, note = _normalize_evidence_refs(draft)
    assert normalized.claims == draft.claims
    assert note is None
