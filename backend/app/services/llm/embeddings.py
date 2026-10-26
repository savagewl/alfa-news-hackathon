import logging
import math
import time
from collections.abc import Sequence
from functools import lru_cache

import httpx2 as httpx

from app.services.llm.config import ServicesSettings, get_services_settings

logger = logging.getLogger("app.services.llm.embeddings")

_CONNECT_TIMEOUT_SECONDS = 5.0


class EmbeddingsClient:
    """Эмбеддинги bge-m3: шлюз (`/embeddings`), при сбое — локальный Ollama."""

    def __init__(
        self,
        settings: ServicesSettings | None = None,
        client: httpx.Client | None = None,
        gateway_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_services_settings()
        self._client = client or httpx.Client(
            base_url=self._settings.embed_ollama_host.rstrip("/"),
            timeout=httpx.Timeout(
                self._settings.embed_timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS
            ),
        )
        self._gateway = gateway_client
        self._gateway_down_until = 0.0

    @property
    def model_version(self) -> str:
        return f"{self._settings.embed_model}:{self._settings.embed_dim}d"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        batches = [list(chunk) for chunk in _batched(texts, self._settings.embed_batch_size)]
        results: list[list[float]] = []
        for batch in batches:
            results.extend(self._embed_batch(batch))
        return results

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def close(self) -> None:
        """Release connections held by a long-lived worker-owned client."""
        self._client.close()
        if self._gateway is not None:
            self._gateway.close()

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        vectors = self._embed_gateway(batch)
        if vectors is not None:
            return [normalize_vector(vector) for vector in vectors]
        return self._embed_ollama(batch)

    def _gateway_client(self) -> httpx.Client | None:
        settings = self._settings
        if not settings.embed_gateway_enabled or settings.llm_mock or not settings.llm_api_base:
            return None
        if time.monotonic() < self._gateway_down_until:
            return None
        if self._gateway is None:
            key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
            self._gateway = httpx.Client(
                base_url=settings.llm_api_base.rstrip("/"),
                headers={"Authorization": f"Bearer {key}"} if key else {},
                timeout=httpx.Timeout(
                    settings.embed_gateway_timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS
                ),
            )
        return self._gateway

    def _embed_gateway(self, batch: list[str]) -> list[list[float]] | None:
        """Векторы со шлюза или None, если шлюз не отработал (тогда идёт Ollama)."""
        gateway = self._gateway_client()
        if gateway is None:
            return None
        settings = self._settings
        try:
            response = gateway.post(
                "/embeddings",
                json={
                    "model": settings.embed_gateway_model or settings.embed_model,
                    "input": batch,
                    "encoding_format": "float",
                },
            )
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda row: row.get("index", 0))
            vectors = [row["embedding"] for row in rows]
            if len(vectors) != len(batch) or any(
                len(vector) != settings.embed_dim for vector in vectors
            ):
                raise ValueError("неожиданное число векторов или размерность")
        except Exception as exc:
            self._gateway_down_until = time.monotonic() + settings.embed_gateway_cooldown_seconds
            logger.warning(
                "Эмбеддинги шлюза недоступны (%s); используется Ollama", type(exc).__name__
            )
            return None
        logger.debug("Эмбеддинги получены со шлюза: %d векторов", len(batch))
        return vectors

    def _embed_ollama(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self._client.post(
                "/api/embed", json={"model": self._settings.embed_model, "input": batch}
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RuntimeError(
                f"Ollama недоступен по {self._settings.embed_ollama_host} "
                f"({type(exc).__name__}). Проверьте, что Ollama запущена и модель "
                f"{self._settings.embed_model} загружена, и значение AI_EMBED_OLLAMA_HOST "
                "(на хосте — http://127.0.0.1:11434, в Docker — host.docker.internal)."
            ) from exc
        response.raise_for_status()
        data = response.json()
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise RuntimeError("Ollama вернул неожиданный формат embeddings")
        normalized = []
        for vector in vectors:
            if len(vector) != self._settings.embed_dim:
                raise ValueError(
                    f"Неожиданная размерность эмбеддинга: {len(vector)} "
                    f"!= {self._settings.embed_dim}"
                )
            normalized.append(normalize_vector(vector))
        logger.debug(
            "Эмбеддинги получены: %d векторов (%s)", len(batch), self._settings.embed_model
        )
        return normalized


def _batched(items: Sequence[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def normalize_vector(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm < 1e-12:
        logger.warning("Получен нулевой вектор — возвращаю без нормализации")
        return list(vector)
    return [component / norm for component in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


@lru_cache
def get_embedder() -> EmbeddingsClient:
    return EmbeddingsClient()
