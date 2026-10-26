"""Set CSV_TEST_POSTGRES_DB to a dedicated migrated PostgreSQL database."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.main import create_app
from app.models import Company, CompanyPayment, CompanyPersonLink, CsvImport, Person
from app.repositories.csv_import import save_payments
from tests.test_csv_import import COMPANY_FIELDS, company_row, csv_bytes


@pytest.fixture
def client():
    database = os.environ.get("CSV_TEST_POSTGRES_DB")
    if not database:
        pytest.skip("Set CSV_TEST_POSTGRES_DB to a dedicated migrated PostgreSQL database")
    settings = Settings(postgres_db=database)
    with TestClient(create_app(settings)) as client:
        email = f"csv-{uuid4().hex}@example.com"
        credentials = {"login": email, "password": "test-csv-password"}
        assert client.post("/user/register", json=credentials).status_code == 201
        token = client.post("/user/login", json=credentials).json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        yield client, settings


def db_rows(settings, statement):
    async def query():
        engine = create_async_engine(settings.database_url, poolclass=NullPool)
        try:
            async with engine.connect() as connection:
                return (await connection.execute(statement)).mappings().all()
        finally:
            await engine.dispose()

    return asyncio.run(query())


def upload(client, kind, content):
    return client.post(f"/imports/{kind}", files={"file": ("dataset.csv", content, "text/csv")})


def create_company(client, company_id):
    result = client.post("/companies", json={"id": company_id, "name": "Тестовая компания"})
    assert result.status_code == 201, result.text
    return result.json()


def test_relationships_payments_placeholders_and_upserts(client):
    http, settings = client
    a, b, person, other_owned = [uuid4().hex for _ in range(4)]
    create_company(http, a)
    create_company(http, other_owned)
    payments = csv_bytes(
        ["PAYER", "BENEFICIARY", "PAY_AMT", "PAY_CNT"],
        [
            {"PAYER": a, "BENEFICIARY": b, "PAY_AMT": "123.45", "PAY_CNT": "2"},
            {"PAYER": a, "BENEFICIARY": b, "PAY_AMT": "6", "PAY_CNT": "1"},
        ],
    )
    result = upload(http, "payments", payments)
    assert result.status_code == 200, result.text
    payment_import = result.json()["import_id"]
    assert result.json()["counts"]["placeholder_companies"] == 1
    repeated = upload(http, "payments", payments).json()
    assert repeated["reused"] is True
    assert repeated["import_id"] == payment_import
    stored = db_rows(settings, select(CompanyPayment).where(CompanyPayment.payer_company_id == a))
    assert len(stored) == 2
    assert stored[0]["payment_amount"] == Decimal("123.45")
    assert stored[0]["beneficiary_company_id"] == b
    assert stored[0]["payment_count"] == 2
    persons = csv_bytes(
        ["SYNTHETIC_F", "SYNTHETIC_U", "CITY_NAME", "RC_NAME"],
        [
            {"SYNTHETIC_F": person, "SYNTHETIC_U": c, "CITY_NAME": "Москва", "RC_NAME": "Москва"}
            for c in [a, other_owned]
        ],
    )
    result = upload(http, "persons", persons)
    assert result.status_code == 200, result.text
    assert result.json()["counts"]["persons"] == 1
    assert result.json()["counts"]["links"] == 2
    assert len(db_rows(settings, select(Person).where(Person.id == person))) == 1
    assert (
        len(
            db_rows(
                settings, select(CompanyPersonLink).where(CompanyPersonLink.person_id == person)
            )
        )
        == 2
    )
    companies = csv_bytes(COMPANY_FIELDS, [company_row(a, CITY="Казань", CL_PRODUCT_LOA_MARK="1")])
    result = upload(http, "companies", companies)
    assert result.status_code == 200, result.text
    company = db_rows(settings, select(Company).where(Company.id == a))[0]
    assert company["is_placeholder"] is False
    assert company["has_credit"] is True
    assert company["city"] == "Казань"
    assert db_rows(settings, select(Company).where(Company.id == b))[0]["is_placeholder"] is True
    assert upload(http, "companies", companies).json()["reused"] is True
    changed = csv_bytes(COMPANY_FIELDS, [company_row(a, CITY="Москва", CL_PRODUCT_LOA_MARK="0")])
    assert upload(http, "companies", changed).status_code == 200
    company = db_rows(settings, select(Company).where(Company.id == a))[0]
    assert company["city"] == "Москва"
    assert company["has_credit"] is False


def test_invalid_file_does_not_persist_first_valid_row(client):
    http, settings = client
    a, b = uuid4().hex, uuid4().hex
    content = f"PAYER,BENEFICIARY,PAY_AMT,PAY_CNT\n{a},{b},10,1\n{a},{b},20,1.5\n".encode()
    response = upload(http, "payments", content)
    assert response.status_code == 422
    assert response.json()["detail"]["errors"][0]["row"] == 3
    assert not db_rows(settings, select(Company).where(Company.id.in_([a, b])))


def test_database_failure_rolls_back_whole_import(client, monkeypatch):
    http, settings = client
    a, b = uuid4().hex, uuid4().hex
    create_company(http, a)
    content = f"PAYER,BENEFICIARY,PAY_AMT,PAY_CNT\n{a},{b},10,1\n".encode()

    async def fail_after_insert(session, rows, import_id):
        await save_payments(session, rows, import_id)
        raise SQLAlchemyError("private failure details")

    monkeypatch.setattr("app.services.csv_import.save_payments", fail_after_insert)
    response = upload(http, "payments", content)
    assert response.status_code == 503
    assert "private failure" not in response.text
    assert not db_rows(settings, select(Company).where(Company.id == b))
    assert len(db_rows(settings, select(Company).where(Company.id == a))) == 1
    assert not db_rows(settings, select(CompanyPayment).where(CompanyPayment.payer_company_id == a))


def test_concurrent_identical_uploads_are_idempotent(client):
    http, settings = client
    a, b = uuid4().hex, uuid4().hex
    create_company(http, a)
    content = f"PAYER,BENEFICIARY,PAY_AMT,PAY_CNT\n{a},{b},10,1\n".encode()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: upload(http, "payments", content), range(2)))
    assert all(result.status_code == 200 for result in results), [r.text for r in results]
    assert sorted(r.json()["reused"] for r in results) == [False, True]
    assert len({r.json()["import_id"] for r in results}) == 1
    assert (
        db_rows(
            settings,
            select(func.count())
            .select_from(CompanyPayment)
            .where(CompanyPayment.payer_company_id == a),
        )[0]["count_1"]
        == 1
    )
    record_id = results[0].json()["import_id"]
    assert len(db_rows(settings, select(CsvImport).where(CsvImport.id == record_id))) == 1


def test_duplicate_companies_keep_last_entire_row_and_warn_on_repeated_import(client):
    http, settings = client
    a, b = uuid4().hex, uuid4().hex
    last = company_row(a, CITY="Казань", REGION="Татарстан", CL_PRODUCT_LOA_MARK="0")
    content = csv_bytes(
        COMPANY_FIELDS,
        [
            company_row(
                a, CITY="Москва", REGION="Москва", CL_PRODUCT_LOA_MARK="1", REVENUE_COMPANY="100"
            ),
            company_row(b, CITY="Самара"),
            company_row(a, CITY="Омск"),
            last,
        ],
    )
    response = upload(http, "companies", content)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["row_count"] == 4
    assert result["counts"]["companies"] == 2
    assert result["counts"]["skipped_duplicate_rows"] == 2
    assert len(result["warnings"]) == 1
    assert "Пропущено 2" in result["warnings"][0]
    assert "последняя строка" in result["warnings"][0]
    stored = db_rows(settings, select(Company).where(Company.id.in_([a, b])))
    assert len(stored) == 2
    company = next(row for row in stored if row["id"] == a)
    assert company["city"] == "Казань"
    assert company["has_credit"] is False
    assert company["annual_revenue"] is None
    assert company["source_data"] == last
    assert http.get(f"/companies/{a}").status_code == 200
    repeated = upload(http, "companies", content).json()
    assert repeated["reused"] is True
    assert repeated["import_id"] == result["import_id"]
    assert repeated["warnings"] == result["warnings"]
    assert repeated["counts"] == result["counts"]


def test_import_without_duplicate_companies_has_no_warning(client):
    http, _ = client
    result = upload(http, "companies", csv_bytes(COMPANY_FIELDS, [company_row(uuid4().hex)])).json()
    assert result["warnings"] == []
    assert result["counts"]["skipped_duplicate_rows"] == 0
