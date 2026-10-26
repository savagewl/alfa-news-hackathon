import pytest

from app.services.llm.config import get_services_settings


@pytest.fixture(autouse=True)
def disable_background_workers(monkeypatch):
    """Ordinary API tests must not contact external services or race database mocks."""
    monkeypatch.setenv("RSS_ENABLED", "false")
    monkeypatch.setenv("ARTICLE_CONTENT_ENABLED", "false")
    monkeypatch.setenv("ANALYTICS_ENABLED", "false")
    monkeypatch.setenv("AI_NEWS_WORKER_ENABLED", "false")
    monkeypatch.setenv("AI_PROFILE_ENABLED", "false")
    monkeypatch.setenv("AI_ASSESSMENT_ENABLED", "false")
    get_services_settings.cache_clear()
    yield
    get_services_settings.cache_clear()
