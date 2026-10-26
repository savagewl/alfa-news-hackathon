from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.config import Settings
from app.db.session import get_session
from app.main import create_app
from app.schemas.company import CompanyListItem, CompanyPaymentResponse


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1E-7", "0.0000001"),
        ("1.2E+5", "120000"),
        ("1E+30", "1000000000000000000000000000000"),
        ("0E-20", "0.00000000000000000000"),
        ("12345678901234567890.12345678901234567890", "12345678901234567890.12345678901234567890"),
        ("123.4500", "123.4500"),
        ("0", "0"),
    ],
)
def test_money_json_uses_plain_decimal_and_preserves_precision(value, expected):
    amount = Decimal(value)
    company = CompanyListItem(
        id="001",
        name=None,
        group_id=None,
        okved_code=None,
        okved_name=None,
        city=None,
        region=None,
        annual_revenue=amount,
        is_placeholder=False,
    )
    payment = CompanyPaymentResponse(
        id=1,
        payer_company_id="001",
        beneficiary_company_id="002",
        payment_amount=amount,
        payment_count=1,
        import_id=uuid4(),
        source_row_number=2,
    )
    assert company.model_dump(mode="json")["annual_revenue"] == expected
    assert payment.model_dump(mode="json")["payment_amount"] == expected
    assert company.model_dump()["annual_revenue"] == amount
    assert isinstance(company.annual_revenue, Decimal)


def test_company_and_payment_http_responses_use_plain_decimals():
    app = create_app(Settings(_env_file=None, rss_enabled=False))
    session = AsyncMock()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=uuid4())
    app.dependency_overrides[get_session] = lambda: session
    fields = {
        field: None
        for field in app.openapi()["components"]["schemas"]["CompanyResponse"]["properties"]
    }
    fields.update(
        id="001",
        name="ООО Пример",
        okved_name="Торговля компьютерами",
        annual_revenue=Decimal("1E-7"),
        is_placeholder=False,
        updated_at=datetime.now(UTC),
    )
    company = SimpleNamespace(**fields)
    payment = SimpleNamespace(
        id=1,
        payer_company_id="001",
        beneficiary_company_id="002",
        payment_amount=Decimal("1E+30"),
        payment_count=1,
        import_id=uuid4(),
        source_row_number=2,
    )
    with TestClient(app) as client:
        session.scalar.return_value = company
        response = client.get("/companies/001")
        assert response.status_code == 200, response.text
        assert response.json()["annual_revenue"] == "0.0000001"
        assert response.json()["okved_name"] == "Торговля компьютерами"
        assert "Торговля компьютерами" in response.text
        session.scalar.side_effect = [company, 1]
        session.scalars.return_value = SimpleNamespace(all=lambda: [payment])
        response = client.get("/companies/001/payments")
        assert response.status_code == 200, response.text
        assert response.json()["items"][0]["payment_amount"] == "1000000000000000000000000000000"
