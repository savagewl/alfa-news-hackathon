import json
import logging
import re
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Protocol

import httpx2 as httpx
from pydantic import BaseModel, ValidationError

from app.services.llm.config import ServicesSettings, get_services_settings

logger = logging.getLogger("app.services.llm.provider")

_LT, _GT = chr(60), chr(62)
_THINK_RE = re.compile(
    _LT + "think" + _GT + r".*?" + _LT + "/think" + _GT, re.DOTALL | re.IGNORECASE
)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*$")


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    usage_available: bool = True


class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        json_mode: bool,
        purpose: str,
    ) -> LLMResponse: ...


class LLMConfigError(RuntimeError):
    """LLM не настроен: нет AI_LLM_API_BASE и не включён явный AI_LLM_MOCK."""


class LLMBudgetExceededError(RuntimeError):
    """Исчерпан суточный бюджет токенов (AI_LLM_DAILY_TOKEN_BUDGET)."""


def _today() -> str:
    # Лимит шлюза сбрасывается в 3:00 МСК = 00:00 UTC.
    return datetime.now(UTC).date().isoformat()


class UsageLedger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = _today()
        self._day_tokens = 0
        self._stats: dict[str, Counter] = defaultdict(Counter)
        self._models: dict[str, Counter] = defaultdict(Counter)
        self._finish_reasons: dict[str, Counter] = defaultdict(Counter)

    def tokens_today(self) -> int:
        with self._lock:
            if self._day != _today():
                return 0
            return self._day_tokens

    def add(self, purpose: str, response: LLMResponse) -> None:
        with self._lock:
            today = _today()
            if self._day != today:
                self._day, self._day_tokens = today, 0
            self._day_tokens += (
                response.prompt_tokens + response.completion_tokens + response.reasoning_tokens
            )
            stats = self._stats[purpose]
            stats["calls"] += 1
            stats["prompt_tokens"] += response.prompt_tokens
            stats["completion_tokens"] += response.completion_tokens
            stats["reasoning_tokens"] += response.reasoning_tokens
            if not response.usage_available:
                stats["usage_unavailable"] += 1
            self._models[purpose][response.model] += 1
            if response.finish_reason:
                self._finish_reasons[purpose][response.finish_reason] += 1

    def report(self) -> dict[str, dict]:
        with self._lock:
            report: dict[str, dict] = {}
            for purpose, stats in self._stats.items():
                report[purpose] = {
                    "calls": stats["calls"],
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "reasoning_tokens": stats["reasoning_tokens"],
                    "models": dict(self._models[purpose]),
                }
                if stats["usage_unavailable"]:
                    report[purpose]["usage_unavailable"] = stats["usage_unavailable"]
                if self._finish_reasons[purpose]:
                    report[purpose]["finish_reasons"] = dict(self._finish_reasons[purpose])
            return report


USAGE_LEDGER = UsageLedger()


class MockProvider:
    DEFAULT_RESPONSES = {
        "news_analysis": (
            '{"summary": "Демо-режим: суммари новости недоступно без LLM API.", '
            '"topics": ["демо"], '
            '"affected_sectors": [], '
            '"regions": ["Россия"], '
            '"impact_kinds": ["other"], '
            '"facts": [], '
            '"size_relevance": ["любой"]}'
        ),
        "news_assessment": (
            '{"brief_impact": "Демо-режим: персонализированная сводка недоступна без LLM API.", '
            '"risks": "Демо-режим: оценка рисков недоступна без LLM API.", '
            '"opportunities": "Демо-режим: оценка возможностей недоступна без LLM API.", '
            '"bank_analytics": null, "bank_refs": []}'
        ),
        "insight_analyst": (
            '{"headline": "Демо-инсайт", '
            '"facts": [], '
            '"bank_position": null, '
            '"ai_interpretation": "Демо-режим: интерпретация недоступна без LLM API.", '
            '"risks": [], "opportunities": [], "suggested_actions": [], '
            '"limitations": ["LLM API не подключён"], '
            '"claims": []}'
        ),
        "insight_verifier": (
            '{"status": "APPROVED", "remarks": [], "missing_evidence": [], "retry_queries": []}'
        ),
        "followup": "Демо-режим: ответ на вопрос недоступен без подключённого LLM API.",
    }

    def __init__(self, handler=None) -> None:
        self._handler = handler

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        json_mode: bool,
        purpose: str,
    ) -> LLMResponse:
        if self._handler is not None:
            content = self._handler(purpose, messages, json_mode)
        else:
            content = self.DEFAULT_RESPONSES.get(purpose, "{}")
        return LLMResponse(content=content, model=model, prompt_tokens=1, completion_tokens=1)


_MAX_ATTEMPTS = 3
_MAX_RETRY_AFTER_SECONDS = 30.0


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """Пауза перед повтором: Retry-After (если есть) или экспоненциальный backoff."""
    backoff = min(0.5 * 2**attempt, 8)
    if response.status_code == 429:
        backoff = min(2.0 * 2**attempt, _MAX_RETRY_AFTER_SECONDS)
    try:
        retry_after = float(response.headers.get("retry-after", ""))
    except ValueError:
        return backoff
    return min(max(retry_after, backoff), _MAX_RETRY_AFTER_SECONDS)


class OpenAICompatProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: float,
        client: httpx.Client | None = None,
        reasoning_budget: int | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
        )
        self._timeout = timeout
        self._reasoning_budget = reasoning_budget

    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        json_mode: bool,
        stream: bool,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if self._reasoning_budget is not None:
            payload["reasoning"] = {"max_token_budget": self._reasoning_budget}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        json_mode: bool,
        purpose: str,
    ) -> LLMResponse:
        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
            stream=False,
        )
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            delay = min(0.5 * 2**attempt, 8)
            try:
                response = self._client.post("/chat/completions", json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
            else:
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = RuntimeError(f"HTTP {response.status_code}")
                    delay = _retry_delay(response, attempt)
                else:
                    response.raise_for_status()
                    data = response.json()
                    choice = data.get("choices", [{}])[0]
                    content = (choice.get("message") or {}).get("content") or ""
                    usage = data.get("usage") or {}
                    details = usage.get("completion_tokens_details") or {}
                    reasoning = int(
                        usage.get("reasoning_tokens") or details.get("reasoning_tokens") or 0
                    )
                    logger.info(
                        "LLM usage: purpose=%s, prompt=%d, completion=%d, reasoning=%d, finish=%s",
                        purpose,
                        int(usage.get("prompt_tokens") or 0),
                        int(usage.get("completion_tokens") or 0),
                        reasoning,
                        choice.get("finish_reason"),
                    )
                    return LLMResponse(
                        content=content.strip(),
                        model=model,
                        prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        completion_tokens=int(usage.get("completion_tokens") or 0),
                        reasoning_tokens=reasoning,
                        finish_reason=choice.get("finish_reason"),
                    )
            if last_error is not None and attempt < _MAX_ATTEMPTS - 1:
                time.sleep(delay)
        raise RuntimeError(f"LLM API недоступен (purpose={purpose}): {last_error}")

    def complete_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        json_mode: bool,
        purpose: str,
        content_callback: Callable[[str], None],
    ) -> LLMResponse:
        """Aggregate one SSE completion while exposing its content fragments."""

        payload = self._payload(
            messages,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
            stream=True,
        )
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            received_content = False
            delay = min(0.5 * 2**attempt, 8)
            timeout = self._timeout
            deadline = getattr(self._client, "deadline", None)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Interpretation deadline exceeded")
                timeout = min(timeout, remaining)
            try:
                with self._client.stream(
                    "POST", "/chat/completions", json=payload, timeout=timeout
                ) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        last_error = RuntimeError(f"HTTP {response.status_code}")
                        delay = _retry_delay(response, attempt)
                    else:
                        response.raise_for_status()
                        parts: list[str] = []
                        usage: dict = {}
                        finish_reason = None
                        for line in response.iter_lines():
                            if deadline is not None and time.monotonic() >= deadline:
                                raise TimeoutError("Interpretation deadline exceeded")
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            event = json.loads(data)
                            if event.get("usage"):
                                usage = event["usage"]
                            for choice in event.get("choices") or []:
                                finish_reason = choice.get("finish_reason") or finish_reason
                                fragment = (choice.get("delta") or {}).get("content")
                                if not isinstance(fragment, str) or not fragment:
                                    continue
                                received_content = True
                                parts.append(fragment)
                                content_callback(fragment)
                        details = usage.get("completion_tokens_details") or {}
                        reasoning = int(
                            usage.get("reasoning_tokens")
                            or details.get("reasoning_tokens")
                            or 0
                        )
                        usage_available = any(
                            key in usage
                            for key in (
                                "prompt_tokens",
                                "completion_tokens",
                                "reasoning_tokens",
                            )
                        )
                        if not usage_available:
                            logger.info("LLM stream usage unavailable: purpose=%s", purpose)
                        return LLMResponse(
                            content="".join(parts).strip(),
                            model=model,
                            prompt_tokens=int(usage.get("prompt_tokens") or 0),
                            completion_tokens=int(usage.get("completion_tokens") or 0),
                            reasoning_tokens=reasoning,
                            finish_reason=finish_reason,
                            usage_available=usage_available,
                        )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if received_content:
                    raise RuntimeError("LLM stream interrupted after partial content") from exc
                last_error = exc
            if last_error is not None and attempt < _MAX_ATTEMPTS - 1:
                time.sleep(delay)
        raise RuntimeError(f"LLM API недоступен (purpose={purpose}): {last_error}")


@lru_cache
def _openai_provider_cached(
    base_url: str, api_key: str | None, timeout: float, reasoning_budget: int | None
) -> OpenAICompatProvider:
    return OpenAICompatProvider(base_url, api_key, timeout, reasoning_budget=reasoning_budget)


def get_llm_provider(settings: ServicesSettings | None = None) -> LLMProvider:
    settings = settings or get_services_settings()
    if settings.llm_mock:
        return MockProvider()
    if not settings.llm_api_base:
        raise LLMConfigError(
            "AI_LLM_API_BASE не задан. Укажите OpenAI-совместимый эндпоинт "
            "или явно включите демо-режим: AI_LLM_MOCK=true."
        )
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    return _openai_provider_cached(
        settings.llm_api_base, api_key, settings.llm_timeout_seconds, settings.llm_reasoning_budget
    )


def strip_think(text: str) -> str:
    cleaned = _THINK_RE.sub("", text)
    lines = [line for line in cleaned.splitlines() if not _FENCE_RE.match(line.strip())]
    return "\n".join(lines).strip()


def parse_json(content: str):
    cleaned = strip_think(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _check_budget(settings: ServicesSettings) -> None:
    budget = settings.llm_daily_token_budget
    if budget is not None and USAGE_LEDGER.tokens_today() >= budget:
        raise LLMBudgetExceededError(
            f"Суточный бюджет токенов исчерпан ({budget}); сброс в 00:00 UTC (3:00 МСК)"
        )


def complete_json(
    purpose: str,
    schema: type[BaseModel],
    messages: list[dict[str, str]],
    *,
    provider: LLMProvider | None = None,
    settings: ServicesSettings | None = None,
    temperature: float | None = None,
    content_callback: Callable[[str], None] | None = None,
    attempt_callback: Callable[[], None] | None = None,
) -> tuple[BaseModel, LLMResponse]:
    settings = settings or get_services_settings()
    provider = provider or get_llm_provider(settings)
    model = settings.model_for(purpose)
    if temperature is None:
        temperature = settings.llm_temperature
    convo = list(messages)
    attempts = settings.llm_max_retries + 1
    last_error: Exception | None = None
    for _ in range(attempts):
        _check_budget(settings)
        if attempt_callback is not None:
            attempt_callback()
        stream_method = getattr(provider, "complete_stream", None)
        if content_callback is not None and callable(stream_method):
            response = stream_method(
                convo,
                model=model,
                temperature=temperature,
                json_mode=True,
                purpose=purpose,
                content_callback=content_callback,
            )
        else:
            response = provider.complete(
                convo, model=model, temperature=temperature, json_mode=True, purpose=purpose
            )
        USAGE_LEDGER.add(purpose, response)
        try:
            return schema.model_validate(parse_json(response.content)), response
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            convo = convo + [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": (
                        "Ответ не прошёл валидацию схемы: "
                        f"{exc}. Верни строго валидный JSON по схеме, без пояснений."
                    ),
                },
            ]
    raise RuntimeError(f"LLM не вернула валидный JSON (purpose={purpose}): {last_error}")


def complete_text(
    purpose: str,
    messages: list[dict[str, str]],
    *,
    provider: LLMProvider | None = None,
    settings: ServicesSettings | None = None,
    temperature: float | None = None,
) -> tuple[str, LLMResponse]:
    settings = settings or get_services_settings()
    provider = provider or get_llm_provider(settings)
    model = settings.model_for(purpose)
    if temperature is None:
        temperature = settings.llm_temperature
    attempts = settings.llm_max_retries + 1
    last_error: str | None = None
    for _ in range(attempts):
        _check_budget(settings)
        response = provider.complete(
            list(messages), model=model, temperature=temperature, json_mode=False, purpose=purpose
        )
        USAGE_LEDGER.add(purpose, response)
        content = strip_think(response.content)
        if content:
            return content, response
        last_error = "пустой ответ модели"
    raise RuntimeError(f"LLM вернула пустой ответ (purpose={purpose}): {last_error}")


def messages(system: str, user: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
