import json

import httpx2 as httpx
import pytest
from pydantic import BaseModel

from app.services.llm.config import ServicesSettings
from app.services.llm.provider import (
    USAGE_LEDGER,
    LLMResponse,
    MockProvider,
    OpenAICompatProvider,
    complete_json,
    complete_text,
    parse_json,
    strip_think,
)
from app.services.llm.schemas import NewsAnalysis


class Dummy(BaseModel):
    summary: str
    count: int = 0


class _StaticProvider:
    def __init__(self, response: LLMResponse) -> None:
        self._response = response

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        json_mode: bool,
        purpose: str,
    ) -> LLMResponse:
        return self._response


def _settings() -> ServicesSettings:
    return ServicesSettings(_env_file=None)


def test_complete_json_validates_schema():
    provider = MockProvider(
        handler=lambda purpose, messages, json_mode: json.dumps({"summary": "ok", "count": 2})
    )
    value, response = complete_json(
        "news_analysis",
        Dummy,
        [{"role": "user", "content": "x"}],
        provider=provider,
        settings=_settings(),
    )
    assert value.summary == "ok"
    assert value.count == 2
    assert response.model == "google/gemma-4-26b-a4b-it"


def test_complete_json_retries_with_feedback():
    responses = ["совсем не json", '```json\n{"summary": "ok", "count": 0}\n```']
    calls: list[list[dict]] = []

    def handler(purpose, messages, json_mode):
        calls.append(messages)
        return responses[len(calls) - 1]

    value, _ = complete_json(
        "news_analysis",
        Dummy,
        [{"role": "user", "content": "x"}],
        provider=MockProvider(handler=handler),
        settings=_settings(),
    )
    assert value.summary == "ok"
    assert len(calls) == 2
    assert calls[1][-1]["role"] == "user"
    assert "валидацию схемы" in calls[1][-1]["content"]


def test_complete_json_fails_after_retries():
    provider = MockProvider(handler=lambda purpose, messages, json_mode: "не json")
    with pytest.raises(RuntimeError):
        complete_json(
            "news_analysis",
            Dummy,
            [{"role": "user", "content": "x"}],
            provider=provider,
            settings=_settings(),
        )


def test_strip_think_and_fences():
    lt, gt = chr(60), chr(62)
    raw = lt + "think" + gt + " рассуждения " + lt + "/think" + gt + '```json\n{"a": 1}\n```'
    cleaned = strip_think(raw)
    assert cleaned == '{"a": 1}'
    assert parse_json(cleaned) == {"a": 1}


def test_usage_ledger_counts_calls():
    before = USAGE_LEDGER.report().get("news_analysis", {}).get("calls", 0)
    complete_json(
        "news_analysis",
        Dummy,
        [{"role": "user", "content": "x"}],
        provider=MockProvider(),
        settings=_settings(),
    )
    after = USAGE_LEDGER.report()["news_analysis"]["calls"]
    assert after == before + 1


def test_complete_text_returns_content():
    answer, _ = complete_text(
        "followup",
        [{"role": "user", "content": "?"}],
        provider=MockProvider(),
        settings=_settings(),
    )
    assert "Демо-режим" in answer


def test_complete_text_retries_on_empty():
    responses = ["", " ответ"]
    calls: list[list[dict]] = []

    def handler(purpose, messages, json_mode):
        calls.append(messages)
        return responses[len(calls) - 1]

    answer, _ = complete_text(
        "followup",
        [{"role": "user", "content": "?"}],
        provider=MockProvider(handler=handler),
        settings=_settings(),
    )
    assert answer == "ответ"
    assert len(calls) == 2


def test_openai_provider_json_mode_and_retries():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"summary": "ok", "count": 1}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")
    provider = OpenAICompatProvider("http://mock", None, 5, client=client)
    value, response = complete_json(
        "news_analysis",
        Dummy,
        [{"role": "user", "content": "x"}],
        provider=provider,
        settings=_settings(),
    )
    assert value.count == 1
    assert response.prompt_tokens == 5
    assert response.completion_tokens == 7
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert payloads[0]["model"] == "google/gemma-4-26b-a4b-it"
    assert "reasoning" not in payloads[0]


def test_openai_provider_streams_json_without_extra_request():
    payloads: list[dict] = []
    fragments = ['{"sum', 'mary":"ok",', '"count":1}']

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        lines = [
            "data: "
            + json.dumps(
                {"choices": [{"delta": {"content": fragment}, "finish_reason": None}]}
            )
            for fragment in fragments
        ]
        lines.extend(
            [
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
        return httpx.Response(200, content=("\n\n".join(lines) + "\n\n").encode())

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")
    provider = OpenAICompatProvider("http://mock", None, 5, client=client)
    streamed: list[str] = []
    attempts: list[bool] = []
    value, response = complete_json(
        "news_analysis",
        Dummy,
        [{"role": "user", "content": "x"}],
        provider=provider,
        settings=_settings(),
        content_callback=streamed.append,
        attempt_callback=lambda: attempts.append(True),
    )
    assert value == Dummy(summary="ok", count=1)
    assert streamed == fragments
    assert attempts == [True]
    assert len(payloads) == 1
    assert payloads[0]["stream"] is True
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert response.finish_reason == "stop"
    assert response.usage_available is False


def test_openai_provider_reasoning_budget_payload():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"summary": "ok", "count": 1}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "reasoning_tokens": 99,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")
    provider = OpenAICompatProvider("http://mock", None, 5, client=client, reasoning_budget=1500)
    value, response = complete_json(
        "news_analysis",
        Dummy,
        [{"role": "user", "content": "x"}],
        provider=provider,
        settings=_settings(),
    )
    assert value.count == 1
    assert payloads[0]["reasoning"] == {"max_token_budget": 1500}
    assert response.finish_reason == "stop"
    assert response.reasoning_tokens == 99


def test_usage_ledger_records_finish_reason():
    response = LLMResponse(
        content='{"summary": "ok", "count": 1}',
        model="test",
        prompt_tokens=1,
        completion_tokens=1,
        reasoning_tokens=7,
        finish_reason="length",
    )
    before = USAGE_LEDGER.report().get("followup", {})
    complete_json(
        "followup",
        Dummy,
        [{"role": "user", "content": "x"}],
        provider=_StaticProvider(response),
        settings=_settings(),
    )
    after = USAGE_LEDGER.report()["followup"]
    assert (
        after["finish_reasons"].get("length", 0)
        == before.get("finish_reasons", {}).get("length", 0) + 1
    )
    assert after["reasoning_tokens"] == before.get("reasoning_tokens", 0) + 7


def test_news_analysis_schema_from_news_analysis():
    payload = {
        "summary": "Суммари новости",
        "topics": ["ставка"],
        "affected_sectors": [{"code": "25.11", "name": "Производство конструкций"}],
        "regions": ["Россия"],
        "impact_kinds": ["rate"],
        "facts": ["Ставка 14%"],
        "size_relevance": ["крупный"],
    }
    analysis = NewsAnalysis.model_validate(payload)
    assert analysis.affected_sectors[0].code == "25.11"


def test_get_llm_provider_requires_explicit_mock():
    from app.services.llm.provider import LLMConfigError, get_llm_provider

    with pytest.raises(LLMConfigError):
        get_llm_provider(ServicesSettings(_env_file=None, llm_api_base=None, llm_mock=False))
    mock = get_llm_provider(ServicesSettings(_env_file=None, llm_mock=True))
    assert isinstance(mock, MockProvider)


def test_openai_provider_honors_retry_after(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("app.services.llm.provider.time.sleep", sleeps.append)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": "limit"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")
    provider = OpenAICompatProvider("http://mock", None, 5, client=client)
    response = provider.complete(
        [{"role": "user", "content": "x"}],
        model="m",
        temperature=0,
        json_mode=False,
        purpose="followup",
    )
    assert response.content == "ok"
    assert sleeps == [7.0]


def test_openai_provider_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr("app.services.llm.provider.time.sleep", lambda _: None)
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(429)),
        base_url="http://mock",
    )
    provider = OpenAICompatProvider("http://mock", None, 5, client=client)
    with pytest.raises(RuntimeError, match="недоступен"):
        provider.complete(
            [{"role": "user", "content": "x"}],
            model="m",
            temperature=0,
            json_mode=False,
            purpose="followup",
        )


def test_daily_token_budget_blocks_calls():
    from app.services.llm.provider import LLMBudgetExceededError

    used = USAGE_LEDGER.tokens_today()
    settings = ServicesSettings(_env_file=None, llm_daily_token_budget=max(used, 1))
    if used == 0:
        complete_text(
            "followup",
            [{"role": "user", "content": "?"}],
            provider=MockProvider(),
            settings=settings,
        )
    with pytest.raises(LLMBudgetExceededError):
        complete_text(
            "followup",
            [{"role": "user", "content": "?"}],
            provider=MockProvider(),
            settings=settings,
        )


def _sse(*events: str) -> bytes:
    return ("\n\n".join(events) + "\n\n").encode()


def test_stream_requests_usage_and_counts_tokens():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=_sse(
                'data: {"choices":[{"delta":{"content":"{\\"summary\\":\\"ok\\"}"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":5}}',
                "data: [DONE]",
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")
    provider = OpenAICompatProvider("http://mock", None, 5, client=client)
    value, response = complete_json(
        "news_analysis",
        Dummy,
        [{"role": "user", "content": "x"}],
        provider=provider,
        settings=_settings(),
        content_callback=lambda _: None,
    )
    assert value.summary == "ok"
    assert payloads[0]["stream_options"] == {"include_usage": True}
    assert response.usage_available is True
    assert (response.prompt_tokens, response.completion_tokens) == (11, 5)


def test_stream_falls_back_when_gateway_rejects_stream_options():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if "stream_options" in payload:
            return httpx.Response(422, json={"error": "unknown field stream_options"})
        return httpx.Response(
            200,
            content=_sse(
                'data: {"choices":[{"delta":{"content":"{\\"summary\\":\\"ok\\"}"}}]}',
                "data: [DONE]",
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")
    provider = OpenAICompatProvider("http://mock", None, 5, client=client)
    value, response = complete_json(
        "news_analysis",
        Dummy,
        [{"role": "user", "content": "x"}],
        provider=provider,
        settings=_settings(),
        content_callback=lambda _: None,
    )
    assert value.summary == "ok" and response.usage_available is False
    assert "stream_options" in payloads[0] and "stream_options" not in payloads[1]
    assert len(payloads) == 2


def test_stream_skips_malformed_sse_lines():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                "data: {broken json",
                'data: ["not", "an", "object"]',
                'data: {"choices":[{"delta":{"content":"{\\"summary\\":\\"ok\\"}"}}]}',
                "data: [DONE]",
            ),
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock")
    provider = OpenAICompatProvider("http://mock", None, 5, client=client)
    value, _ = complete_json(
        "news_analysis",
        Dummy,
        [{"role": "user", "content": "x"}],
        provider=provider,
        settings=_settings(),
        content_callback=lambda _: None,
    )
    assert value.summary == "ok"
