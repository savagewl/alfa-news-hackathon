from types import SimpleNamespace
from uuid import uuid4

from app.schemas.chat import ChatAnswerClaim, ChatAnswerPayload, ChatSource
from app.services.chat import (
    MAX_HISTORY_CHARS,
    MAX_HISTORY_MESSAGES,
    _validate_payload,
    bounded_history,
)


def message(role: str, content: str, *, status: str = "ready"):
    return SimpleNamespace(id=uuid4(), role=role, content=content, status=status)


def test_bounded_history_uses_latest_ready_messages_and_character_budget():
    messages = [message("user", f"old-{index}") for index in range(20)]
    messages.append(message("assistant", "ignored", status="failed"))
    history = bounded_history(messages)
    assert "old-0" not in history
    assert "old-19" in history
    assert history.count("Пользователь:") == MAX_HISTORY_MESSAGES

    huge = bounded_history([message("user", "x" * (MAX_HISTORY_CHARS * 2))])
    assert len(huge) <= MAX_HISTORY_CHARS


def test_structured_claims_require_exact_known_refs_and_documentary_facts():
    news_ref = "news:https://example.test/a"
    allowed = {
        news_ref: ChatSource(
            evidence_ref=news_ref,
            kind="news",
            title="Новость",
            source_url="https://example.test/a",
        )
    }
    payload = ChatAnswerPayload(
        answer="Факт",
        claims=[ChatAnswerClaim(text="Факт", kind="fact", evidence_refs=[news_ref])],
    )
    kind, sources, warnings = _validate_payload(payload, allowed)
    assert kind == "fact" and sources[0].evidence_ref == news_ref and warnings == []

    invented = payload.model_copy(
        update={
            "claims": [
                ChatAnswerClaim(
                    text="Факт",
                    kind="fact",
                    evidence_refs=[news_ref + "/invented"],
                )
            ]
        }
    )
    # выдуманная ссылка отбрасывается, тезис без подтверждения не попадает в источники
    kind, sources, warnings = _validate_payload(invented, allowed)
    assert sources == [] and kind == "ai_inference"
    assert any("не подтверждена" in warning for warning in warnings)

    # пропущенный префикс и опечатка в ссылке исправляются
    sloppy = payload.model_copy(
        update={
            "claims": [
                ChatAnswerClaim(text="Факт", kind="fact", evidence_refs=["https://example.test/a"])
            ]
        }
    )
    kind, sources, warnings = _validate_payload(sloppy, allowed)
    assert kind == "fact" and sources[0].evidence_ref == news_ref and warnings == []


def test_bank_citation_does_not_turn_ai_inference_into_bank_position():
    bank_ref = f"bank:{uuid4()}#p1"
    allowed = {
        bank_ref: ChatSource(
            evidence_ref=bank_ref,
            kind="bank",
            title="Прогноз банка",
            source_url="/analytics/file#page=1",
        )
    }
    inference = ChatAnswerPayload(
        answer="Для этой компании возможен риск.",
        claims=[
            ChatAnswerClaim(
                text="Для этой компании возможен риск.",
                kind="ai_inference",
                evidence_refs=[bank_ref],
            )
        ],
    )
    kind, _, _ = _validate_payload(inference, allowed)
    assert kind == "ai_inference"

    unsupported_position = inference.model_copy(
        update={
            "claims": [
                ChatAnswerClaim(
                    text="Позиция банка",
                    kind="bank_position",
                    evidence_refs=["company:Регион"],
                )
            ]
        }
    )
    allowed["company:Регион"] = ChatSource(
        evidence_ref="company:Регион", kind="company", title="Регион"
    )
    kind, sources, warnings = _validate_payload(unsupported_position, allowed)
    assert kind == "ai_inference"
    assert sources[0].evidence_ref == "company:Регион"
    assert any("Позиция банка" in warning for warning in warnings)

    fact_from_profile = unsupported_position.model_copy(
        update={
            "claims": [
                ChatAnswerClaim(text="Регион", kind="fact", evidence_refs=["company:Регион"])
            ]
        }
    )
    kind, _, warnings = _validate_payload(fact_from_profile, allowed)
    assert kind == "ai_inference" and any("Факт без документа" in w for w in warnings)


def test_missing_evidence_answer_may_have_no_claims_and_unsupported_payload_only_warns():
    kind, sources, warnings = _validate_payload(
        ChatAnswerPayload(
            answer="Данных недостаточно.", claims=[], missing_evidence=["Нет периода платежей"]
        ),
        {},
    )
    assert kind == "ai_inference" and sources == [] and warnings == []
    kind, sources, warnings = _validate_payload(ChatAnswerPayload(answer="Не знаю"), {})
    assert kind == "ai_inference" and sources == []
    assert any("не подкреплён" in warning for warning in warnings)


def test_inline_source_markers_are_removed_from_answer_text():
    from app.services.chat import strip_inline_refs

    text = (
        "Новость касается экспорта [news:https://www.kommersant.ru/doc/1]. "
        "Для вашей компании [company:Отрасль (ОКВЭД 25.11)] это опосредованно "
        "[bank:3f2c#p1]."
    )
    assert strip_inline_refs(text) == (
        "Новость касается экспорта. Для вашей компании это опосредованно."
    )
    assert strip_inline_refs("Без ссылок [1]") == "Без ссылок [1]"


def test_strip_inline_refs_keeps_punctuation():
    from app.services.chat import strip_inline_refs

    assert strip_inline_refs("Итог , ясно !") == "Итог, ясно!"


def test_parenthesized_markers_with_nested_parentheses_are_removed():
    from app.services.chat import strip_inline_refs

    text = (
        "Для компании (company:Отрасль (ОКВЭД 25.11)), в квартале есть риск "
        "(news:https://www.kommersant.ru/doc/9044417). Итог (не ссылка) остаётся."
    )
    assert strip_inline_refs(text) == (
        "Для компании, в квартале есть риск. Итог (не ссылка) остаётся."
    )


def test_markers_with_introductory_words_in_parentheses_are_removed():
    from app.services.chat import strip_inline_refs

    text = (
        "Прогноз экономиста (источник: news:https://www.klerk.ru/buh/news/714841/), "
        "ставка останется 14% (см. bank:3f2c#p1). Подробности (в таблице) ниже."
    )
    assert strip_inline_refs(text) == (
        "Прогноз экономиста, ставка останется 14%. Подробности (в таблице) ниже."
    )


def test_bank_passage_numbers_are_removed_from_chat_answer():
    from app.services.chat import strip_inline_refs

    assert strip_inline_refs("Банк ждёт 14% [B1], инфляция 6% [B2, B3].") == (
        "Банк ждёт 14%, инфляция 6%."
    )
