from uuid import uuid4

import pytest

from tests.test_csv_import import COMPANY_FIELDS, company_row, csv_bytes
from tests.test_csv_import_integration import client as client


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp1251"])
def test_csv_text_and_money_survive_database_and_http_response(client, encoding):
    http, _ = client
    company_id, counterparty = uuid4().hex, uuid4().hex
    name = "Торговля компьютерами, ООО «Пример»"
    content = csv_bytes(
        COMPANY_FIELDS,
        [company_row(company_id, MAINOKVED_NAME=name, CITY="Москва", REVENUE_COMPANY="1E-7")],
        encoding=encoding,
        delimiter=";",
    )
    imported = http.post(
        f"/imports/companies?encoding={encoding}",
        files={"file": ("companies.csv", content, "text/csv")},
    )
    assert imported.status_code == 200, imported.text
    company = http.get(f"/companies/{company_id}").json()
    assert company["name"] is None  # This anonymized CSV has no company-name column.
    assert company["okved_name"] == name
    assert company["city"] == "Москва"
    assert company["source_data"]["MAINOKVED_NAME"] == name
    assert company["annual_revenue"] == "0.0000001"
    listed = http.get("/companies", params={"q": company_id}).json()["items"]
    assert len(listed) == 1
    assert listed[0]["okved_name"] == name
    assert listed[0]["annual_revenue"] == "0.0000001"
    payments = csv_bytes(
        ["PAYER", "BENEFICIARY", "PAY_AMT", "PAY_CNT"],
        [
            {"PAYER": company_id, "BENEFICIARY": counterparty, "PAY_AMT": "1E-9", "PAY_CNT": "1"},
            {"PAYER": company_id, "BENEFICIARY": counterparty, "PAY_AMT": "1E+30", "PAY_CNT": "1"},
        ],
    )
    result = http.post("/imports/payments", files={"file": ("payments.csv", payments, "text/csv")})
    assert result.status_code == 200, result.text
    page = http.get(f"/companies/{company_id}/payments").json()
    assert [row["payment_amount"] for row in page["items"]] == [
        "0.000000001",
        "1000000000000000000000000000000",
    ]
