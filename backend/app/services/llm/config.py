from functools import lru_cache
from uuid import UUID

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServicesSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    llm_api_base: str | None = None
    llm_api_key: SecretStr | None = None
    llm_mock: bool = False
    llm_timeout_seconds: float = Field(default=120, ge=1)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_news_temperature: float = Field(default=0.0, ge=0, le=2)
    # Бюджет reasoning-токенов (OpenRouter `reasoning.max_token_budget`); None = не отправлять.
    # Reasoning-модели без этого параметра тратят десятки секунд «мыслей» на каждый вызов.
    llm_reasoning_budget: int | None = Field(default=None, ge=1)

    # Мягкий суточный лимит токенов (prompt+completion+reasoning); None = без лимита.
    llm_daily_token_budget: int | None = Field(default=None, ge=1)
    # Максимум символов новости, отправляемых в LLM (защита от огромных content_text).
    news_max_chars: int = Field(default=12_000, ge=500)

    llm_model: str = "google/gemma-4-26b-a4b-it"

    embed_ollama_host: str = "http://127.0.0.1:11434"
    embed_model: str = "bge-m3"
    embed_dim: int = Field(default=1024, ge=1)
    embed_timeout_seconds: float = Field(default=300, ge=1)
    embed_batch_size: int = Field(default=16, ge=1)
    # Сначала POST {AI_LLM_API_BASE}/embeddings; при любой ошибке — Ollama.
    # Модель шлюза должна быть тем же bge-m3: версия векторов в БД от неё не зависит.
    # По умолчанию выключено: у ключей хакатона bge-m3 на шлюзе недоступен (403).
    embed_gateway_enabled: bool = False
    embed_gateway_model: str | None = None  # None = AI_EMBED_MODEL
    embed_gateway_timeout_seconds: float = Field(default=20, ge=1)
    # После сбоя шлюза он не опрашивается столько секунд (сразу идёт Ollama).
    embed_gateway_cooldown_seconds: float = Field(default=60, ge=0)

    # Реранкер шлюза (POST {AI_LLM_API_BASE}/rerank). None = выключен; при сбое работает
    # прежняя формула «косинус + веса правил».
    rerank_model: str | None = None
    rerank_timeout_seconds: float = Field(default=5, ge=0.5)
    rerank_cooldown_seconds: float = Field(default=60, ge=0)
    rerank_top_k: int = Field(default=40, ge=1, le=200)
    rerank_cache_seconds: float = Field(default=600, ge=0)

    relevance_cosine_weight: float = Field(default=0.4, ge=0, le=1)
    relevance_threshold: float = Field(default=0.35, ge=0, le=1)
    insight_max_cycles: int = Field(default=2, ge=1, le=5)
    interpretation_timeout_seconds: int = Field(default=90, ge=5, le=300)
    interpretation_concurrency: int = Field(default=2, ge=1, le=8)
    interpretation_cache_hours: int = Field(default=24, ge=1, le=168)
    # Only explicitly selected case documents may be attributed to the bank.
    bank_document_ids: list[UUID] = Field(default_factory=list)

    news_worker_enabled: bool = True
    news_worker_poll_seconds: float = Field(default=2.0, ge=0.2, le=60)
    news_worker_batch_size: int = Field(default=8, ge=1, le=100)
    news_worker_concurrency: int = Field(default=2, ge=1, le=8)
    news_worker_lease_seconds: int = Field(default=300, ge=30, le=3600)
    news_worker_max_attempts: int = Field(default=5, ge=1, le=20)
    news_worker_retry_seconds: int = Field(default=30, ge=1, le=3600)
    news_worker_config_retry_seconds: int = Field(default=300, ge=10, le=3600)

    profile_enabled: bool = True
    profile_poll_seconds: float = Field(default=2.0, ge=0.2, le=60)
    profile_batch_size: int = Field(default=8, ge=1, le=100)
    profile_retry_seconds: int = Field(default=30, ge=1, le=3600)
    profile_lease_seconds: int = Field(default=300, ge=30, le=3600)

    # Короткая оценка новости для компании (риски, возможности, аналитика банка): один вызов LLM,
    # готовится фоном для новостей, попавших на открытую страницу ленты.
    assessment_enabled: bool = True
    assessment_poll_seconds: float = Field(default=2.0, ge=0.2, le=60)
    assessment_batch_size: int = Field(default=4, ge=1, le=50)
    assessment_concurrency: int = Field(default=4, ge=1, le=8)
    assessment_lease_seconds: int = Field(default=180, ge=30, le=3600)
    assessment_max_attempts: int = Field(default=3, ge=1, le=10)
    assessment_retry_seconds: int = Field(default=30, ge=1, le=3600)
    assessment_cache_hours: int = Field(default=24, ge=1, le=168)
    # Сколько новостей страницы ленты ставится в очередь на оценку за один запрос.
    assessment_enqueue_limit: int = Field(default=20, ge=1, le=100)
    assessment_bank_passages: int = Field(default=3, ge=1, le=8)

    _PURPOSES = frozenset(
        {
            "news_analysis",
            "news_assessment",
            "insight_analyst",
            "insight_revise",
            "insight_verifier",
            "followup",
        }
    )

    def model_for(self, purpose: str) -> str:
        if purpose not in self._PURPOSES:
            raise ValueError(f"Неизвестный purpose: {purpose}")
        return self.llm_model


@lru_cache
def get_services_settings() -> ServicesSettings:
    return ServicesSettings()
