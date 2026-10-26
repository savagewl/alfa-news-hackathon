import json

from app.services.llm.config import ServicesSettings
from app.services.llm.news_analysis import _normalize, analyze_news_text, process_news_text
from app.services.llm.provider import MockProvider
from app.services.llm.schemas import NewsAnalysis

NEWS = (
    "Банк России сохранил ключевую ставку на уровне 14% на заседании 11 сентября 2026 года. "
    "Риторика регулятора стала жёстче, что может отсрочить удешевление кредитов для бизнеса."
)


def _news_handler(purpose, messages, json_mode):
    assert purpose == "news_analysis"
    assert "Новость:" in messages[-1]["content"]
    return json.dumps(
        {
            "summary": "Банк России сохранил ключевую ставку на уровне 14%, риторика стала жёстче.",
            "topics": ["ключевая ставка"],
            "affected_sectors": [{"code": None, "name": "финансы"}],
            "regions": ["Россия"],
            "impact_kinds": ["rate", "совершенно неизвестный тип"],
            "facts": ["Ставка сохранена на уровне 14%"],
            "size_relevance": ["крупный", "любой"],
        },
        ensure_ascii=False,
    )


def test_analyze_news_normalizes_impacts():
    analysis = analyze_news_text(NEWS, llm=MockProvider(handler=_news_handler))
    assert "14%" in analysis.summary
    assert analysis.impact_kinds == ["rate"]
    assert analysis.affected_sectors[0].name == "финансы"


def test_analyze_news_empty_impacts_falls_back_to_other():
    def handler(purpose, messages, json_mode):
        payload = json.loads(_news_handler(purpose, messages, json_mode))
        payload["impact_kinds"] = []
        return json.dumps(payload, ensure_ascii=False)

    analysis = analyze_news_text(NEWS, llm=MockProvider(handler=handler))
    assert analysis.impact_kinds == ["other"]


def test_normalize_empty_size_relevance_falls_back_to_any():
    analysis = NewsAnalysis(summary="Событие без указания масштаба", size_relevance=[])
    normalized = _normalize(analysis)
    assert normalized.size_relevance == ["любой"]


def test_process_news_text_builds_vector(fake_embedder):
    processed = process_news_text(
        NEWS,
        url="https://example.ru/rate",
        llm=MockProvider(handler=_news_handler),
        embedder=fake_embedder,
    )
    assert processed.url == "https://example.ru/rate"
    assert "14%" in processed.summary
    assert len(processed.vector) == 8
    assert abs(sum(value * value for value in processed.vector) - 1.0) < 1e-9
    assert processed.embedding_model_version == "fake-embedder:8d"
    assert fake_embedder.calls == [[processed.summary]]


def test_process_news_text_uses_real_settings_default():
    settings = ServicesSettings(_env_file=None)
    assert settings.llm_model == "google/gemma-4-26b-a4b-it"
    purposes = (
        "news_analysis",
        "insight_analyst",
        "insight_revise",
        "insight_verifier",
        "followup",
    )
    for purpose in purposes:
        assert settings.model_for(purpose) == settings.llm_model
