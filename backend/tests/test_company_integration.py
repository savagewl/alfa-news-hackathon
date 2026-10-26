from uuid import uuid4

from tests.test_csv_import import COMPANY_FIELDS, company_row, csv_bytes
from tests.test_csv_import_integration import client as client
from tests.test_csv_import_integration import create_company, upload


def test_company_profile_list_filters_and_pagination(client):
    http, _ = client
    prefix, group = uuid4().hex, uuid4().hex
    a, b = f"{prefix}-001", f"{prefix}-002"
    content = csv_bytes(
        COMPANY_FIELDS,
        [
            company_row(
                a,
                GROUP_ID=group,
                MAINOKVED_CODE="01.01",
                MAINOKVED_NAME="Производство мебели",
                REVENUE_COMPANY="15000000000.50",
                CITY="Казань",
                REGION="Татарстан",
                CL_PRODUCT_LOA_MARK="1",
                CL_PRODUCT_DEP_MARK="0",
            ),
            company_row(b, GROUP_ID=group, CITY="Москва", REGION="Москва"),
        ],
    )
    imported = upload(http, "companies", content).json()
    response = http.get(f"/companies/{a}")
    assert response.status_code == 200, response.text
    profile = response.json()
    assert profile["id"] == a
    assert profile["okved_code"] == "01.01"
    assert profile["annual_revenue"] == "15000000000.50"
    assert profile["has_credit"] is True
    assert profile["has_deposit"] is False
    assert profile["has_leasing"] is None
    assert profile["source_data"]["SYNTHETIC_U"] == a
    assert profile["last_import_id"] == imported["import_id"]
    assert profile["is_placeholder"] is False
    assert profile["updated_at"]
    page = http.get("/companies", params={"group_id": group, "limit": 1, "offset": 1}).json()
    assert page["total"] == 2
    assert page["limit"] == 1
    assert page["offset"] == 1
    assert [c["id"] for c in page["items"]] == [b]
    assert "source_data" not in page["items"][0]
    filtered = http.get(
        "/companies", params={"group_id": group, "q": "каз", "region": "Татарстан"}
    ).json()
    assert [c["id"] for c in filtered["items"]] == [a]
    assert (
        http.get("/companies", params={"group_id": group, "is_placeholder": True}).json()["total"]
        == 0
    )
    assert http.get("/companies", params={"group_id": group, "offset": 100}).json()["items"] == []
    # % and _ must be literal characters, not SQL LIKE wildcards.
    assert http.get("/companies", params={"group_id": group, "q": "%"}).json()["total"] == 0
    assert http.get("/companies", params={"group_id": group, "q": "_"}).json()["total"] == 0


def test_company_person_links_placeholders_and_empty_lists(client):
    http, _ = client
    a, b, c, person = [uuid4().hex for _ in range(4)]
    create_company(http, a)
    create_company(http, b)
    content = csv_bytes(
        ["SYNTHETIC_F", "SYNTHETIC_U", "CITY_NAME", "RC_NAME"],
        [
            {
                "SYNTHETIC_F": person,
                "SYNTHETIC_U": company,
                "CITY_NAME": "Москва",
                "RC_NAME": "Москва",
            }
            for company in [a, b]
        ],
    )
    imported = upload(http, "persons", content).json()
    profile = http.get(f"/companies/{a}").json()
    assert profile["is_placeholder"] is False
    assert profile["has_credit"] is None
    for company in [a, b]:
        response = http.get(f"/companies/{company}/persons")
        assert response.status_code == 200, response.text
        assert response.json()["total"] == 1
        assert response.json()["items"] == [
            {
                "person_id": person,
                "city": "Москва",
                "region": "Москва",
                "person_import_id": imported["import_id"],
                "link_import_id": imported["import_id"],
            }
        ]
    assert http.get(f"/companies/{a}/persons?offset=1").json()["items"] == []
    assert upload(http, "companies", csv_bytes(COMPANY_FIELDS, [company_row(c)])).status_code == 200
    for suffix in ["persons", "payments", "payment-imports"]:
        response = http.get(f"/companies/{c}/{suffix}")
        assert response.status_code == 200
        assert response.json()["items"] == []
        assert response.json()["total"] == 0


def test_payment_direction_self_payments_snapshots_and_pagination(client):
    http, _ = client
    a, b, c = [uuid4().hex for _ in range(3)]
    create_company(http, a)
    create_company(http, b)
    first = csv_bytes(
        ["PAYER", "BENEFICIARY", "PAY_AMT", "PAY_CNT"],
        [
            {"PAYER": a, "BENEFICIARY": b, "PAY_AMT": "10.25", "PAY_CNT": "2"},
            {"PAYER": b, "BENEFICIARY": a, "PAY_AMT": "20", "PAY_CNT": "3"},
            {"PAYER": a, "BENEFICIARY": a, "PAY_AMT": "30", "PAY_CNT": "4"},
            {"PAYER": c, "BENEFICIARY": b, "PAY_AMT": "40", "PAY_CNT": "5"},
        ],
    )
    first_id = upload(http, "payments", first).json()["import_id"]
    second = csv_bytes(
        ["PAYER", "BENEFICIARY", "PAY_AMT", "PAY_CNT"],
        [
            {"PAYER": a, "BENEFICIARY": b, "PAY_AMT": "99.99", "PAY_CNT": "1"},
        ],
    )
    second_id = upload(http, "payments", second).json()["import_id"]
    response = http.get(f"/companies/{a}/payments")
    assert response.status_code == 200, response.text
    all_rows = response.json()
    assert all_rows["total"] == 4
    assert all_rows["import_id"] is None
    assert all_rows["direction"] == "all"
    assert all_rows["items"][0]["payment_amount"] == "10.25"
    assert all_rows["items"][0]["payment_count"] == 2
    assert {row["import_id"] for row in all_rows["items"]} == {first_id, second_id}
    assert len({row["id"] for row in all_rows["items"]}) == 4
    incoming = http.get(f"/companies/{a}/payments", params={"direction": "incoming"}).json()
    assert incoming["total"] == 2
    assert all(row["beneficiary_company_id"] == a for row in incoming["items"])
    outgoing = http.get(f"/companies/{a}/payments", params={"direction": "outgoing"}).json()
    assert outgoing["total"] == 3
    assert all(row["payer_company_id"] == a for row in outgoing["items"])
    selected = http.get(f"/companies/{a}/payments", params={"import_id": first_id}).json()
    assert selected["total"] == 3
    assert selected["import_id"] == first_id
    assert {row["import_id"] for row in selected["items"]} == {first_id}
    page = http.get(f"/companies/{a}/payments", params={"limit": 2, "offset": 1}).json()
    assert page["total"] == 4
    assert page["items"] == all_rows["items"][1:3]
    assert (
        http.get(f"/companies/{a}/payments", params={"import_id": str(uuid4())}).json()["total"]
        == 0
    )
    imports = http.get(f"/companies/{a}/payment-imports").json()
    assert imports["total"] == 2
    assert [item["import_id"] for item in imports["items"]] == [second_id, first_id]
    assert imports["items"][1]["row_count"] == 4
    assert imports["items"][1]["company_payment_rows"] == 3
    page = http.get(f"/companies/{a}/payment-imports", params={"limit": 1, "offset": 1}).json()
    assert page["total"] == 2
    assert page["items"][0]["import_id"] == first_id
