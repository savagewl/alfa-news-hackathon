from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Alfa News API"
    debug: bool = False
    analytics_enabled: bool = True
    analytics_max_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    analytics_max_pages: int = Field(default=100, ge=1, le=500)
    analytics_parse_timeout_seconds: int = Field(default=900, ge=10, le=3600)
    analytics_ocr_dpi: int = Field(default=200, ge=100, le=300)
    analytics_ocr_languages: str = "rus+eng"
    analytics_tessdata: str | None = None
    analytics_embeddings_enabled: bool = True
    rss_enabled: bool = True
    article_content_enabled: bool = True
    article_content_concurrency: int = Field(default=2, ge=1, le=8)
    article_content_poll_seconds: float = Field(default=2, ge=1, le=60)
    rss_concurrency: int = Field(default=4, ge=1, le=16)
    rss_request_timeout_seconds: int = Field(default=30, ge=1, le=60)
    rss_max_response_bytes: int = Field(default=5 * 1024 * 1024, ge=1024, le=20 * 1024 * 1024)
    rss_lease_seconds: int = Field(default=120, ge=90, le=600)
    rss_snapshot_retention_days: int = Field(default=7, ge=1, le=90)
    rss_run_retention_days: int = Field(default=30, ge=1, le=365)
    csv_import_max_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    csv_import_max_rows: int = Field(default=100_000, ge=1)
    jwt_secret: SecretStr | None = Field(default=None, min_length=32)
    jwt_expire_minutes: int = Field(default=1440, ge=1)
    postgres_user: str = "alfa_news"
    postgres_password: SecretStr = SecretStr("alfa_news_local")
    postgres_db: str = "alfa_news"
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)

    @field_validator("analytics_max_bytes")
    @classmethod
    def ensure_analytics_upload_minimum(cls, value: int) -> int:
        # Existing deployments may still supply the former 25 MiB limit via .env.
        return max(value, 50 * 1024 * 1024)

    @property
    def database_url(self) -> URL:
        return URL.create(
            "postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
