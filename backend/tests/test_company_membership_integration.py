from uuid import UUID, uuid4

import jwt
import pytest
from sqlalchemy import select

from app.models import Company, CsvImport, Person
from tests.test_csv_import import COMPANY_FIELDS, company_row, csv_bytes
from tests.test_csv_import_integration import client as client
from tests.test_csv_import_integration import create_company, db_rows, upload


def register_user(http):
    credentials = {"login": f"member-{uuid4().hex}@example.com", "password": "membership-test-pass"}
    response = http.post("/user/register", json=credentials)
    assert response.status_code == 201
    token = http.post("/user/login", json=credentials).json()["access_token"]
    return response.json()["userId"], {"Authorization": f"Bearer {token}"}


def current_user_id(http):
    token = http.headers["Authorization"].removeprefix("Bearer ")
    return jwt.decode(token, options={"verify_signature": False})["sub"]


def upload_as(http, kind, content, headers):
    return http.post(
        f"/imports/{kind}",
        headers=headers,
        files={"file": ("dataset.csv", content, "text/csv")},
    )


def test_create_company_grants_only_creator_access_and_rejects_claims(client):
    http, _ = client
    creator_id = current_user_id(http)
    outsider_id, outsider_headers = register_user(http)
    result = http.post("/companies", json={"name": "ООО Пример", "has_credit": True})
    assert result.status_code == 201, result.text
    company = result.json()
    assert UUID(company["id"])
    assert company["name"] == "ООО Пример"
    assert company["has_credit"] is True
    assert company["is_placeholder"] is False
    assert company["source_data"] is None
    company_id = company["id"]
    assert http.get(f"/companies/{company_id}").status_code == 200
    assert http.get("/companies", params={"q": "ООО Пример"}).json()["total"] == 1
    assert http.get("/companies", headers=outsider_headers).json()["total"] == 0
    members = http.get(f"/companies/{company_id}/users").json()
    assert members["total"] == 1
    assert members["items"][0]["user_id"] == creator_id
    assert set(members["items"][0]) == {"user_id", "full_name", "joined_at"}
    for suffix in ["", "/persons", "/payments", "/payment-imports", "/users"]:
        response = http.get(f"/companies/{company_id}{suffix}", headers=outsider_headers)
        assert response.status_code == 404, response.text
    denied = http.post(
        f"/companies/{company_id}/users",
        json={"user_id": outsider_id},
        headers=outsider_headers,
    )
    assert denied.status_code == 404
    conflict = http.post(
        "/companies",
        json={"id": company_id, "name": "Попытка присвоения"},
        headers=outsider_headers,
    )
    assert conflict.status_code == 409
    assert http.get("/companies", headers=outsider_headers).json()["total"] == 0


def test_many_users_many_companies_and_idempotent_add(client):
    http, _ = client
    a, b = uuid4().hex, uuid4().hex
    create_company(http, a)
    create_company(http, b)
    second_id, second_headers = register_user(http)
    third_id, third_headers = register_user(http)
    personal = http.post("/companies", json={"name": "Личная"}, headers=second_headers).json()["id"]
    for company in [a, b]:
        first = http.post(f"/companies/{company}/users", json={"user_id": second_id})
        assert first.status_code == 200, first.text
        assert first.json()["added_by_user_id"] == current_user_id(http)
        repeated = http.post(f"/companies/{company}/users", json={"user_id": second_id})
        assert repeated.json() == first.json()
        assert http.get(f"/companies/{company}/users").json()["total"] == 2
        assert http.get(f"/companies/{company}", headers=second_headers).status_code == 200
    visible = http.get("/companies", headers=second_headers).json()
    assert visible["total"] == 3
    assert {c["id"] for c in visible["items"]} == {a, b, personal}
    assert http.get("/companies").json()["total"] == 2
    # No roles: any existing member can add another registered user.
    response = http.post(
        f"/companies/{a}/users",
        json={"user_id": third_id},
        headers=second_headers,
    )
    assert response.status_code == 200
    assert http.get(f"/companies/{a}/users").json()["total"] == 3
    assert http.get(f"/companies/{a}", headers=third_headers).status_code == 200
    assert http.get(f"/companies/{b}", headers=third_headers).status_code == 404
    unknown = http.post(f"/companies/{a}/users", json={"user_id": str(uuid4())})
    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "User not found"}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"name": " "},
        {"name": "A", "id": " "},
        {"name": "A", "annual_revenue": "-1"},
        {"name": "A", "user_id": str(uuid4())},
        {"name": "A", "is_placeholder": True},
    ],
)
def test_create_company_validates_input_and_disallows_membership_injection(client, body):
    http, _ = client
    assert http.post("/companies", json=body).status_code == 422
    assert http.get("/companies").json()["total"] == 0


def test_csv_company_import_grants_new_membership_but_cannot_update_foreign_company(client):
    http, settings = client
    own, new = uuid4().hex, uuid4().hex
    content = csv_bytes(COMPANY_FIELDS, [company_row(own, CITY="Казань")])
    result = upload(http, "companies", content)
    assert result.status_code == 200, result.text
    assert http.get(f"/companies/{own}").status_code == 200
    record = db_rows(settings, select(CsvImport).where(CsvImport.id == result.json()["import_id"]))[
        0
    ]
    assert str(record["imported_by_user_id"]) == current_user_id(http)
    _, outsider_headers = register_user(http)
    # Even the already processed, byte-identical CSV must pass access checks.
    assert upload_as(http, "companies", content, outsider_headers).status_code == 403
    mixed = csv_bytes(COMPANY_FIELDS, [company_row(new), company_row(own, CITY="Москва")])
    response = upload_as(http, "companies", mixed, outsider_headers)
    assert response.status_code == 403
    assert not db_rows(settings, select(Company).where(Company.id == new))
    assert http.get(f"/companies/{own}").json()["city"] == "Казань"
    assert http.get("/companies", headers=outsider_headers).json()["total"] == 0


def test_imports_do_not_grant_counterparty_access_or_allow_foreign_links(client):
    http, settings = client
    own, other, unknown1, unknown2 = [uuid4().hex for _ in range(4)]
    create_company(http, own)
    content = f"PAYER,BENEFICIARY,PAY_AMT,PAY_CNT\n{own},{other},10,1\n".encode()
    assert upload(http, "payments", content).status_code == 200
    assert http.get(f"/companies/{own}/payments").json()["total"] == 1
    assert http.get(f"/companies/{other}").status_code == 404
    assert (
        upload(http, "companies", csv_bytes(COMPANY_FIELDS, [company_row(other)])).status_code
        == 403
    )
    assert http.post("/companies", json={"id": other, "name": "Claim"}).status_code == 409
    unknown = f"PAYER,BENEFICIARY,PAY_AMT,PAY_CNT\n{unknown1},{unknown2},10,1\n".encode()
    assert upload(http, "payments", unknown).status_code == 403
    assert not db_rows(settings, select(Company).where(Company.id.in_([unknown1, unknown2])))
    person = uuid4().hex
    foreign = f"SYNTHETIC_F,SYNTHETIC_U,CITY_NAME,RC_NAME\n{person},{other},M,R\n".encode()
    assert upload(http, "persons", foreign).status_code == 403
    assert not db_rows(settings, select(Person).where(Person.id == person))


def test_shared_person_cannot_be_modified_through_only_one_company(client):
    http, _ = client
    own, other, person = [uuid4().hex for _ in range(3)]
    create_company(http, own)
    _, outsider_headers = register_user(http)
    assert (
        http.post(
            "/companies",
            json={"id": other, "name": "Другая"},
            headers=outsider_headers,
        ).status_code
        == 201
    )
    first = f"SYNTHETIC_F,SYNTHETIC_U,CITY_NAME,RC_NAME\n{person},{own},M,R\n".encode()
    second = f"SYNTHETIC_F,SYNTHETIC_U,CITY_NAME,RC_NAME\n{person},{other},M,R\n".encode()
    assert upload(http, "persons", first).status_code == 200
    assert upload_as(http, "persons", second, outsider_headers).status_code == 200
    changed = f"SYNTHETIC_F,SYNTHETIC_U,CITY_NAME,RC_NAME\n{person},{own},Changed,R\n".encode()
    assert upload(http, "persons", changed).status_code == 403
    assert http.get(f"/companies/{own}/persons").json()["items"][0]["city"] == "M"
    assert (
        http.get(f"/companies/{other}/persons", headers=outsider_headers).json()["items"][0]["city"]
        == "M"
    )
