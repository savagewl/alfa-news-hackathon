import csv
import io
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.config import Settings
from app.db.session import get_session
from app.main import create_app
from app.schemas.csv_import import ImportCounts, ImportResponse
from app.services.csv_parser import COMPANY_FIELDS, CsvValidationError, parse_csv


def csv_bytes(headers, rows, *, delimiter=",", encoding="utf-8-sig"):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=headers, delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode(encoding)


def company_row(company_id="001", **changes):
    return {**dict.fromkeys(COMPANY_FIELDS, ""), "SYNTHETIC_U": company_id, **changes}


def test_company_fields_preserve_ids_categories_and_unknowns():
    content = csv_bytes(
        COMPANY_FIELDS,
        [
            company_row(
                MAINOKVED_CODE="01.01",
                COMPANY_AGE_YEARS_GRADATION="5–10 лет",
                AMOUNT_WORKERS_FNS_GRADATION="100–500",
                CL_PRODUCT_LOA_MARK="1.0",
                CL_PRODUCT_DEP_MARK="0",
                REVENUE_COMPANY="15 000 000 000,50",
                NEGATIVEMARK="",
                FTS_COUNTRY_NAME="Китай",
            )
        ],
        delimiter=";",
    )
    parsed = parse_csv(content, "companies", "utf-8-sig", None, 100)
    row = parsed.rows[0]
    assert parsed.delimiter == ";"
    assert row["id"] == "001"
    assert row["okved_code"] == "01.01"
    assert row["age_category"] == "5–10 лет"
    assert row["employees_fns_category"] == "100–500"
    assert row["has_credit"] is True
    assert row["has_deposit"] is False
    assert row["key_owner_flag"] is None
    assert row["annual_revenue"] == Decimal("15000000000.50")
    assert row["source_data"]["FTS_COUNTRY_NAME"] == "Китай"


def test_cp1251_tab_and_multiple_company_links():
    content = csv_bytes(
        ["SYNTHETIC_F", "SYNTHETIC_U", "CITY_NAME", "RC_NAME"],
        [
            {"SYNTHETIC_F": "001", "SYNTHETIC_U": c, "CITY_NAME": "Москва", "RC_NAME": "Москва"}
            for c in ["001", "002"]
        ],
        encoding="cp1251",
        delimiter="\t",
    )
    assert len(parse_csv(content, "persons", "cp1251", None, 10).rows) == 2
    with pytest.raises(CsvValidationError, match="CSV validation failed"):
        parse_csv(content, "persons", "utf-8-sig", None, 10)


@pytest.mark.parametrize(
    "amount,count",
    [
        ("-1", "1"),
        ("Infinity", "1"),
        ("NaN", "1"),
        ("1", "1.5"),
        ("1", str(2**63)),
        ("1", ""),
        ("", "1"),
        ("1e999", "1"),
    ],
)
def test_invalid_payment_values_are_rejected(amount, count):
    content = csv_bytes(
        ["PAYER", "BENEFICIARY", "PAY_AMT", "PAY_CNT"],
        [{"PAYER": "a", "BENEFICIARY": "b", "PAY_AMT": amount, "PAY_CNT": count}],
    )
    with pytest.raises(CsvValidationError) as error:
        parse_csv(content, "payments", "utf-8-sig", None, 10)
    assert error.value.errors[0]["row"] == 2


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"PAYER,PAYER\na,b\n",
        b"PAYER,BENEFICIARY\na,b\n",
        b"PAYER,BENEFICIARY,PAY_AMT,PAY_CNT\na,b,1\n",
        b"PAYER,BENEFICIARY,PAY_AMT,PAY_CNT\na,b,1,1,extra\n",
        b'PAYER,BENEFICIARY,PAY_AMT,PAY_CNT\n"unclosed,b,1,1',
        b"PAYER,BENEFICIARY,PAY_AMT,PAY_CNT\n",
        b"PAYER,BENEFICIARY,PAY_AMT,PAY_CNT\na,b,1,\x00\n",
    ],
)
def test_invalid_structure_is_rejected(content):
    with pytest.raises(CsvValidationError):
        parse_csv(content, "payments", "utf-8-sig", None, 10)


def test_repeated_company_ids_are_accepted_and_input_rows_still_count_toward_limit():
    content = csv_bytes(COMPANY_FIELDS, [company_row(), company_row(CITY="Москва")])
    parsed = parse_csv(content, "companies", "utf-8-sig", None, 10)
    assert len(parsed.rows) == 2
    assert parsed.skipped_duplicate_rows == 1
    assert parsed.rows[-1]["city"] == "Москва"
    with pytest.raises(CsvValidationError):
        parse_csv(content, "companies", "utf-8-sig", None, 1)


def test_identical_repeated_company_rows_are_reported():
    content = csv_bytes(COMPANY_FIELDS, [company_row(), company_row(), company_row()])
    parsed = parse_csv(content, "companies", "utf-8-sig", None, 10)
    assert parsed.skipped_duplicate_rows == 2


def test_conflicting_person_attributes_are_not_silently_overwritten():
    content = b"SYNTHETIC_F,SYNTHETIC_U,CITY_NAME,RC_NAME\nf,a,One,R\nf,b,Two,R\n"
    with pytest.raises(CsvValidationError) as error:
        parse_csv(content, "persons", "utf-8-sig", None, 10)
    assert "Conflicting geography" in error.value.errors[0]["message"]


@pytest.fixture
def client():
    app = create_app(Settings(_env_file=None, csv_import_max_bytes=1024))
    session = AsyncMock()
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as client:
        yield client, session


@pytest.mark.parametrize("kind", ["companies", "persons", "payments"])
def test_upload_requires_authentication(client, kind):
    http, _ = client
    assert http.post(f"/imports/{kind}", files={"file": ("x.csv", b"x")}).status_code == 401


def test_upload_rejects_oversize_and_reports_csv_errors_without_writes(client):
    http, session = client
    http.app.dependency_overrides[get_current_user] = lambda: object()
    assert http.post("/imports/payments", files={"file": ("x.csv", b"x" * 1025)}).status_code == 413
    response = http.post("/imports/payments", files={"file": ("x.csv", b"wrong\n1\n")})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_csv"
    assert "PAYER" in response.json()["detail"]["errors"][0]["columns"]
    session.execute.assert_not_awaited()
    assert http.post("/imports/payments").status_code == 422


def test_import_openapi_exposes_multipart_and_auth(client):
    http, _ = client
    schema = http.get("/openapi.json").json()
    for kind in ["companies", "persons", "payments"]:
        operation = schema["paths"][f"/imports/{kind}"]["post"]
        assert "multipart/form-data" in operation["requestBody"]["content"]
        assert operation["security"]


def test_company_upload_accepts_50_mib_with_multipart_overhead(monkeypatch):
    size = 50 * 1024 * 1024
    content = csv_bytes(COMPANY_FIELDS, [company_row(CITY="a" * 100_000)] * 500)
    content += b"\n" * (size - len(content))
    assert len(content) == size
    importer = AsyncMock(return_value=ImportResponse(
        import_id=uuid4(), kind="companies", filename="companies.csv", row_count=500,
        encoding="utf-8-sig", delimiter=",", reused=False,
        counts=ImportCounts(companies=1, skipped_duplicate_rows=499),
    ))
    monkeypatch.setattr("app.api.imports.import_csv", importer)
    app = create_app(Settings(_env_file=None))
    session = AsyncMock()
    app.dependency_overrides[get_session] = lambda: session
    user = type("User", (), {"user_id": uuid4()})()
    app.dependency_overrides[get_current_user] = lambda: user
    with TestClient(app) as http:
        response = http.post(
            "/imports/companies", files={"file": ("companies.csv", content, "text/csv")}
        )
    assert response.status_code == 200
    assert response.json()["row_count"] == 500
    importer.assert_awaited_once()
    assert importer.await_args.args[3] == content
    assert len(importer.await_args.args[2].rows) == 500
