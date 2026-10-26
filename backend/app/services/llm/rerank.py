import hashlib
import logging
import math
import threading
import time
from collections.abc import Sequence

import httpx2 as httpx

from app.services.llm.config import ServicesSettings, get_services_settings

logger = logging.getLogger("app.services.llm.rerank")

_CONNECT_TIMEOUT_SECONDS = 5.0
_MAX_QUERY_CHARS = 4000
_MAX_DOCUMENT_CHARS = 1500
_MAX_CACHE_ITEMS = 5000


class RerankClient:
    """Реранкер шлюза (`POST /rerank`). Любой сбой даёт None — вызывающий код остаётся на
    прежней формуле «косинус + веса правил»."""

    def __init__(
        self,
        settings: ServicesSettings | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_services_settings()
        self._client = client
        self._down_until = 0.0
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str, str], tuple[float, float]] = {}

    @property
    def enabled(self) -> bool:
        settings = self._settings
        return bool(settings.rerank_model and settings.llm_api_base and not settings.llm_mock)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def rerank(self, query: str, documents: Sequence[str]) -> list[float | None]:
        """Оценки 0..1 по порядку documents; None — оценки нет (реранкер недоступен)."""
        scores: list[float | None] = [None] * len(documents)
        if not documents or not self.enabled:
            return scores
        query = query[:_MAX_QUERY_CHARS]
        trimmed = [text[:_MAX_DOCUMENT_CHARS] for text in documents]
        model = self._settings.rerank_model or ""
        query_key = _digest(model, query)
        now = time.monotonic()
        missing: list[int] = []
        for position, text in enumerate(trimmed):
            cached = self._cache.get((model, query_key, _digest(text)))
            if cached is not None and cached[1] > now:
                scores[position] = cached[0]
            else:
                missing.append(position)
        if not missing or now < self._down_until:
            return scores
        try:
            fresh = self._request(query, [trimmed[position] for position in missing])
        except Exception as exc:
            self._down_until = now + self._settings.rerank_cooldown_seconds
            logger.warning(
                "Реранкер недоступен (%s); используется косинус и веса правил", type(exc).__name__
            )
            return scores
        expires = time.monotonic() + self._settings.rerank_cache_seconds
        with self._lock:
            if len(self._cache) > _MAX_CACHE_ITEMS:
                self._cache.clear()
            for position, score in zip(missing, fresh, strict=True):
                scores[position] = score
                self._cache[(model, query_key, _digest(trimmed[position]))] = (score, expires)
        return scores

    def _http(self) -> httpx.Client:
        if self._client is None:
            settings = self._settings
            key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
            self._client = httpx.Client(
                base_url=(settings.llm_api_base or "").rstrip("/"),
                headers={"Authorization": f"Bearer {key}"} if key else {},
                timeout=httpx.Timeout(
                    settings.rerank_timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS
                ),
            )
        return self._client

    def _request(self, query: str, documents: list[str]) -> list[float]:
        response = self._http().post(
            "/rerank",
            json={
                "model": self._settings.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            },
        )
        response.raise_for_status()
        results = response.json()["results"]
        raw = [0.0] * len(documents)
        seen = set()
        for row in results:
            index = int(row["index"])
            if not 0 <= index < len(documents) or index in seen:
                raise ValueError("реранкер вернул неверный индекс")
            seen.add(index)
            raw[index] = float(row["relevance_score"])
        if len(seen) != len(documents) or not all(math.isfinite(value) for value in raw):
            raise ValueError("реранкер вернул неполный ответ")
        return _to_unit_interval(raw)


def _to_unit_interval(values: list[float]) -> list[float]:
    # Cohere-совместимые реранкеры отдают 0..1, bge-reranker через vLLM/TEI — иногда логиты.
    if all(0.0 <= value <= 1.0 for value in values):
        return values
    return [1.0 / (1.0 + math.exp(-max(min(value, 30.0), -30.0))) for value in values]


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:24]


_shared: RerankClient | None = None


def get_reranker(settings: ServicesSettings | None = None) -> RerankClient:
    """Один клиент на экземпляр настроек: кэш оценок и пауза после сбоя общие для запросов."""
    global _shared
    settings = settings or get_services_settings()
    if _shared is None or _shared._settings is not settings:
        _shared = RerankClient(settings)
    return _shared
