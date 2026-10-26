from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.db.session import get_session
from app.main import create_app
from app.models.analytics import BankAnalyticsSelection
from app.services.analytics.selection import effective_bank_settings
from app.services.llm.config import ServicesSettings, get_services_settings


@pytest.fixture
def bank_client():
    app = create_app()
    session = AsyncMock()
    session.get.return_value = None
    settings = ServicesSettings(_env_file=None, bank_document_ids=[uuid4()])
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_current_user] = lambda: object()
    app.dependency_overrides[get_services_settings] = lambda: settings
    with TestClient(app) as client:
        yield client, session, settings


def test_bank_selection_requires_authentication(bank_client):
    client, session, _ = bank_client
    client.app.dependency_overrides.pop(get_current_user)
    assert client.get("/analytics/bank-documents").status_code == 401
    assert client.put("/analytics/bank-documents", json={"document_ids": []}).status_code == 401
    session.execute.assert_not_awaited()


def test_env_fallback_and_explicit_disable(bank_client):
    client, session, settings = bank_client
    assert client.get("/analytics/bank-documents").json() == {
        "document_ids": list(map(str, settings.bank_document_ids))
    }
    response = client.put("/analytics/bank-documents", json={"document_ids": []})
    assert response.status_code == 200
    assert response.json() == {"document_ids": []}
    session.commit.assert_awaited_once()
    statement = session.execute.await_args.args[0]
    assert statement.compile().params["document_ids"] == []
    session.get.return_value = BankAnalyticsSelection(id=1, document_ids=[])
    assert client.get("/analytics/bank-documents").json() == {"document_ids": []}


def test_apply_processed_documents_and_deduplicate(bank_client):
    client, session, settings = bank_client
    document_id = uuid4()
    session.scalars.return_value = Mock(all=lambda: [SimpleNamespace(
        id=document_id, status="parsed", chunk_count=2, published_at=date(2026, 9, 1)
    )])
    response = client.put(
        "/analytics/bank-documents", json={"document_ids": [str(document_id)] * 2}
    )
    assert response.status_code == 200
    assert response.json() == {"document_ids": [str(document_id)]}
    statement = session.execute.await_args.args[0]
    assert statement.compile().params["document_ids"] == [str(document_id)]
    assert settings.bank_document_ids != [document_id]
    session.commit.assert_awaited_once()


@pytest.mark.parametrize("status,chunks,published", [
    ("queued", 0, date(2026, 9, 1)),
    ("failed", 0, date(2026, 9, 1)),
    ("ready", 2, None),
    ("parsed", 0, date(2026, 9, 1)),
])
def test_unusable_documents_do_not_replace_selection(bank_client, status, chunks, published):
    client, session, _ = bank_client
    document_id = uuid4()
    session.scalars.return_value = Mock(all=lambda: [SimpleNamespace(
        id=document_id, status=status, chunk_count=chunks, published_at=published
    )])
    response = client.put(
        "/analytics/bank-documents", json={"document_ids": [str(document_id)]}
    )
    assert response.status_code == 409
    session.execute.assert_not_awaited()
    session.commit.assert_not_awaited()


def test_unknown_document_and_invalid_uuid(bank_client):
    client, session, _ = bank_client
    session.scalars.return_value = Mock(all=lambda: [])
    assert client.put(
        "/analytics/bank-documents", json={"document_ids": [str(uuid4())]}
    ).status_code == 404
    assert client.put(
        "/analytics/bank-documents", json={"document_ids": ["invalid"]}
    ).status_code == 422
    session.commit.assert_not_awaited()


def test_effective_settings_are_fresh_without_mutating_cached_env():
    import asyncio

    async def check():
        settings = ServicesSettings(_env_file=None, bank_document_ids=[uuid4()])
        original = list(settings.bank_document_ids)
        selected = uuid4()
        session = AsyncMock()
        session.get.return_value = BankAnalyticsSelection(id=1, document_ids=[str(selected)])
        assert (await effective_bank_settings(session, settings)).bank_document_ids == [selected]
        session.get.return_value = BankAnalyticsSelection(id=1, document_ids=[])
        assert (await effective_bank_settings(session, settings)).bank_document_ids == []
        assert settings.bank_document_ids == original

    asyncio.run(check())
