import json

import httpx2 as httpx

from app.services.llm.config import ServicesSettings
from app.services.llm.matching import score_relevance
from app.services.llm.rerank import RerankClient, _to_unit_interval
from app.services.llm.schemas import (
    CompanyFields,
    CompanyProfile,
    CompanyProfileText,
    NewsAnalysis,
    ProcessedNews,
)


def _settings(**overrides) -> ServicesSettings:
    return ServicesSettings(
        _env_file=None,
        llm_api_base="https://gateway.test/v1",
        llm_api_key="secret",
        rerank_model="bge-reranker-v2-m3",
        **overrides,
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://gateway.test/v1")


def _scores(request: httpx.Request, values: list[float]) -> httpx.Response:
    payload = json.loads(request.content)
    assert payload["model"] == "bge-reranker-v2-m3"
    assert payload["top_n"] == len(payload["documents"])
    rows = [{"index": index, "relevance_score": values[index]} for index in range(len(values))]
    return httpx.Response(200, json={"results": list(reversed(rows))})


def test_rerank_returns_scores_in_document_order():
    client = RerankClient(
        _settings(), client=_client(lambda request: _scores(request, [0.9, 0.2, 0.5]))
    )
    assert client.rerank("запрос", ["a", "b", "c"]) == [0.9, 0.2, 0.5]


def test_rerank_disabled_without_model_gateway_or_in_mock():
    for settings in (
        ServicesSettings(_env_file=None, llm_api_base="https://gateway.test/v1"),
        ServicesSettings(_env_file=None, rerank_model="m"),
        _settings(llm_mock=True),
    ):
        client = RerankClient(settings, client=_client(lambda request: httpx.Response(500)))
        assert not client.enabled
        assert client.rerank("q", ["a"]) == [None]


def test_rerank_failure_returns_none_and_pauses():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503)

    client = RerankClient(_settings(), client=_client(handler))
    assert client.rerank("q", ["a", "b"]) == [None, None]
    assert client.rerank("q", ["c"]) == [None]
    assert len(calls) == 1


def test_rerank_rejects_incomplete_or_invalid_responses():
    bodies = [
        {"results": [{"index": 0, "relevance_score": 0.5}]},
        {"results": [{"index": 5, "relevance_score": 0.5}, {"index": 1, "relevance_score": 0.5}]},
        {"results": [{"index": 0, "relevance_score": float("nan")}]},
        {},
    ]
    for body in bodies:
        client = RerankClient(
            _settings(), client=_client(lambda request, body=body: httpx.Response(200, json=body))
        )
        assert client.rerank("q", ["a", "b"]) == [None, None]


def test_rerank_caches_scores_between_calls():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["documents"])
        return _scores(request, [0.7] * len(calls[-1]))

    client = RerankClient(_settings(), client=_client(handler))
    client.rerank("q", ["a", "b"])
    assert client.rerank("q", ["b", "c"]) == [0.7, 0.7]
    assert calls == [["a", "b"], ["c"]]
    client.rerank("другой запрос", ["a"])
    assert calls[-1] == ["a"]


def test_logits_are_squashed_to_unit_interval():
    assert _to_unit_interval([0.2, 0.9]) == [0.2, 0.9]
    squashed = _to_unit_interval([-3.0, 0.0, 4.0])
    assert all(0.0 < value < 1.0 for value in squashed)
    assert squashed == sorted(squashed) and squashed[1] == 0.5


def _profile_and_news():
    profile = CompanyProfile(
        text=CompanyProfileText(synthetic_u="U1", main_text="профиль", aspects={}),
        vectors={"main": [1.0, 0.0]},
        embedding_model_version="bge-m3:2d",
    )
    news = ProcessedNews(
        url="https://example.test/1",
        text="текст",
        summary="суть",
        markup=NewsAnalysis(summary="суть"),
        vector=[0.0, 1.0],
        embedding_model_version="bge-m3:2d",
    )
    return profile, CompanyFields(synthetic_u="U1"), news


def test_rerank_replaces_cosine_term_in_score():
    profile, fields, news = _profile_and_news()
    settings = ServicesSettings(_env_file=None)
    base = score_relevance(profile, fields, news, settings=settings)
    reranked = score_relevance(profile, fields, news, settings=settings, rerank=0.8)
    assert base.cosine == reranked.cosine == 0.0
    assert base.score == 0.0
    assert abs(reranked.score - settings.relevance_cosine_weight * 0.8) < 1e-9
    assert any("реранкер" in reason for reason in reranked.reasons)
