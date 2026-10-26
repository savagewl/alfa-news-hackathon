"""Run against a dedicated migrated DB selected by AUTH_TEST_POSTGRES_DB."""

import os
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def client():
    database = os.environ.get("AUTH_TEST_POSTGRES_DB")
    if not database:
        pytest.skip("Set AUTH_TEST_POSTGRES_DB to a dedicated migrated PostgreSQL database")
    settings = Settings(postgres_db=database)
    with TestClient(create_app(settings)) as client:
        yield client


def test_user_authentication_lifecycle(client):
    email = f"auth-{uuid4().hex}@example.com"
    password = "initial-password"
    response = client.post(
        "/user/register",
        json={"login": email.upper(), "password": password, "fullName": "Иван Иванов"},
    )
    assert response.status_code == 201, response.text
    profile = response.json()
    user_id = profile["userId"]
    assert UUID(user_id).version == 4
    assert profile == {"userId": user_id, "login": email, "fullName": "Иван Иванов"}
    assert (
        client.post("/user/register", json={"login": email, "password": password}).status_code
        == 409
    )
    assert client.post("/user/login", json={"login": email, "password": "wrong"}).status_code == 401
    assert (
        client.post(
            "/user/login", json={"login": "nobody@example.com", "password": "wrong"}
        ).status_code
        == 401
    )

    response = client.post("/user/login", json={"login": email.upper(), "password": password})
    assert response.status_code == 200, response.text
    assert response.json()["expires_in"] == 86400
    headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    assert client.get(f"/user/{user_id}").status_code == 401
    assert (
        client.get(f"/user/{user_id}", headers={"Authorization": "Bearer invalid"}).status_code
        == 401
    )
    assert client.get(f"/user/{user_id}", headers=headers).json() == profile
    assert client.get(f"/user/{uuid4()}", headers=headers).status_code == 403
    reset = {"userId": user_id, "oldPassword": password, "newPassword": "updated-password"}
    assert client.post("/user/reset-password", json=reset).status_code == 401
    assert (
        client.post(
            "/user/reset-password", json={**reset, "userId": str(uuid4())}, headers=headers
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/user/reset-password", json={**reset, "oldPassword": "wrong"}, headers=headers
        ).status_code
        == 400
    )
    assert client.post("/user/reset-password", json=reset, headers=headers).status_code == 204
    assert client.get(f"/user/{user_id}", headers=headers).status_code == 401
    assert (
        client.post("/user/login", json={"login": email, "password": password}).status_code == 401
    )
    response = client.post("/user/login", json={"login": email, "password": "updated-password"})
    assert response.status_code == 200
    new_headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
    assert client.get(f"/user/{user_id}", headers=new_headers).json() == profile


def test_registration_validation_and_optional_name(client):
    assert (
        client.post(
            "/user/register", json={"login": "invalid", "password": "long-password"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/user/register", json={"login": "user@example.com", "password": "short"}
        ).status_code
        == 422
    )
    response = client.post(
        "/user/register", json={"login": f"{uuid4().hex}@example.com", "password": "long-password"}
    )
    assert response.status_code == 201
    assert response.json()["fullName"] is None
