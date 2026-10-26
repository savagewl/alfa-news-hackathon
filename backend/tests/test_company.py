from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import get_current_user
from app.core.config import Settings
from app.db.session import get_session
from app.main import create_app


@pytest.fixture
def client():
    app = create_app(Settings(_env_file=None))
    session = AsyncMock()
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as http:
        yield http, session


@pytest.mark.parametrize(
    "path",
    [
        "/companies",
        "/companies/001",
        "/companies/001/persons",
        "/companies/001/payments",
        "/companies/001/payment-imports",
        "/companies/001/users",
    ],
)
def test_company_routes_require_auth(client, path):
    http, _ = client
    assert http.get(path).status_code == 401


@pytest.mark.parametrize("suffix", ["", "/persons", "/payments", "/payment-imports", "/users"])
def test_unknown_company_returns_404(client, suffix):
    http, session = client
    http.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=uuid4())
    session.scalar.return_value = None
    response = http.get(f"/companies/unknown{suffix}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Company not found"}


@pytest.mark.parametrize(
    "error", [SQLAlchemyError("private-details"), OSError("private-details"), TimeoutError()]
)
def test_company_database_failure_is_sanitized(client, error):
    http, session = client
    http.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=uuid4())
    session.scalar.side_effect = error
    response = http.get("/companies")
    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}


@pytest.mark.parametrize(
    "path",
    [
        "/companies?limit=0",
        "/companies?limit=101",
        "/companies?offset=-1",
        "/companies?q=",
        "/companies/001/payments?direction=invalid",
        "/companies/001/payments?import_id=invalid",
    ],
)
def test_company_query_validation(client, path):
    http, session = client
    http.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=uuid4())
    assert http.get(path).status_code == 422
    session.get.assert_not_awaited()
    session.scalar.assert_not_awaited()


def test_company_openapi_documents_models_filters_and_errors(client):
    http, _ = client
    schema = http.get("/openapi.json").json()
    for path in [
        "/companies",
        "/companies/{company_id}",
        "/companies/{company_id}/persons",
        "/companies/{company_id}/payments",
        "/companies/{company_id}/payment-imports",
        "/companies/{company_id}/users",
    ]:
        operation = schema["paths"][path]["get"]
        assert operation["tags"] == ["Companies"]
        assert operation["security"]
        assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        assert "401" in operation["responses"]
        assert "503" in operation["responses"]
        if "{company_id}" in path:
            assert "404" in operation["responses"]
    parameters = schema["paths"]["/companies/{company_id}/payments"]["get"]["parameters"]
    assert {p["name"] for p in parameters} == {
        "company_id",
        "direction",
        "import_id",
        "limit",
        "offset",
    }
    assert schema["components"]["schemas"]["CompanyResponse"]["properties"]["source_data"]
    assert "201" in schema["paths"]["/companies"]["post"]["responses"]
    assert "409" in schema["paths"]["/companies"]["post"]["responses"]
    assert schema["paths"]["/companies/{company_id}/users"]["post"]["security"]


@pytest.mark.parametrize(
    "path,body",
    [
        ("/companies", {"name": "Company"}),
        ("/companies/001/users", {"user_id": str(uuid4())}),
    ],
)
def test_company_mutations_require_authentication(client, path, body):
    http, _ = client
    assert http.post(path, json=body).status_code == 401
