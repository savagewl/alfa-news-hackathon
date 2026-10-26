from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings
from app.db.session import get_session
from app.main import create_app


@pytest.fixture
def client():
    app = create_app(Settings(_env_file=None))
    with TestClient(app) as client:
        yield client


def test_health_does_not_require_database(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_checks_database(client):
    session = AsyncMock()
    client.app.dependency_overrides[get_session] = lambda: session

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}
    session.execute.assert_awaited_once()
    assert str(session.execute.call_args.args[0]) == "SELECT 1"


@pytest.mark.parametrize("error", [SQLAlchemyError("secret"), OSError("secret"), TimeoutError()])
def test_readiness_returns_503_without_exposing_details(client, error):
    session = AsyncMock()
    session.execute.side_effect = error
    client.app.dependency_overrides[get_session] = lambda: session

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}


def test_database_url_handles_special_characters():
    settings = Settings(_env_file=None, postgres_password="p@ss:/%word")
    assert settings.database_url.password == "p@ss:/%word"


def test_swagger_assets_are_available_locally(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert "cdn.jsdelivr.net" not in response.text
    for asset in ("swagger-ui-bundle.js", "swagger-ui.css", "favicon-32x32.png"):
        asset_url = f"/docs/assets/{asset}"
        assert asset_url in response.text
        asset_response = client.get(asset_url)
        assert asset_response.status_code == 200
        assert asset_response.content
    assert client.get("/", follow_redirects=False).headers["location"].endswith("/docs")


def test_openapi_documents_database_unavailable_response(client):
    schema = client.get("/openapi.json").json()
    readiness = schema["paths"]["/health/ready"]["get"]
    error_schema = readiness["responses"]["503"]["content"]["application/json"]["schema"]
    assert error_schema["$ref"] == "#/components/schemas/UnavailableResponse"
    assert schema["components"]["schemas"]["UnavailableResponse"]["properties"]["detail"]
    assert "/docs" not in schema["paths"]
