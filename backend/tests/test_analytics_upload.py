from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.config import Settings
from app.db.session import get_session
from app.main import create_app
from app.schemas.analytics import DocumentResponse
from tests.test_analytics import sample_pdf


@pytest.mark.parametrize("extra_bytes,expected_status", [(0, 202), (1, 413)])
@pytest.mark.parametrize("env_limit", [None, "26214400"])
def test_analytics_upload_50_mib_boundary(monkeypatch, extra_bytes, expected_status, env_limit):
    if env_limit is None:
        monkeypatch.delenv("ANALYTICS_MAX_BYTES", raising=False)
    else:
        monkeypatch.setenv("ANALYTICS_MAX_BYTES", env_limit)
    limit = 50 * 1024 * 1024
    settings = Settings(_env_file=None)
    assert settings.analytics_max_bytes == limit
    original = sample_pdf()
    content = original + b"\n" * (limit + extra_bytes - len(original))
    record = DocumentResponse(
        id=uuid4(), sha256="a" * 64, filename="research.pdf", title="research",
        publisher="Bank", published_at=None, date_origin="unknown", topics=[],
        created_at=datetime.now(UTC), status="queued", page_count=0, chunk_count=0,
        parser_version=None, warnings=[], error=None, embedding_error=None,
        embedding_model_version=None,
    )
    enqueue = AsyncMock(return_value=(record, False))
    monkeypatch.setattr("app.api.analytics.enqueue", enqueue)
    app = create_app(settings)
    app.dependency_overrides[get_session] = lambda: AsyncMock()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=uuid4())
    with TestClient(app) as client:
        response = client.post(
            "/analytics/documents",
            files={"file": ("research.pdf", content, "application/pdf")},
            data={"publisher": "Bank"},
        )
    assert response.status_code == expected_status
    if expected_status == 202:
        enqueue.assert_awaited_once()
        assert enqueue.await_args.kwargs["content"] == content
        assert response.json()["document"]["status"] == "queued"
    else:
        enqueue.assert_not_awaited()


@pytest.mark.parametrize("configured_mib,effective_mib", [(25, 50), (50, 50), (100, 100)])
def test_analytics_upload_limit_from_existing_env_file(tmp_path, configured_mib, effective_mib):
    env_file = tmp_path / ".env"
    env_file.write_text(f"ANALYTICS_MAX_BYTES={configured_mib * 1024 * 1024}\n")
    assert Settings(_env_file=env_file).analytics_max_bytes == effective_mib * 1024 * 1024
