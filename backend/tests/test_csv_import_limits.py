import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.config import Settings
from app.db.session import get_session
from app.main import create_app
from app.repositories.csv_import import save_persons
from app.schemas.csv_import import ImportCounts, ImportResponse
from app.services.csv_import import response
from app.services.csv_parser import parse_csv


def test_payments_accept_more_than_100000_rows_with_legacy_generic_limit(monkeypatch):
    monkeypatch.setenv("CSV_IMPORT_MAX_ROWS", "100000")
    settings = Settings(_env_file=None)
    assert settings.csv_import_payments_max_rows == 2_000_000
    content = b"PAYER;BENEFICIARY;PAY_AMT;PAY_CNT\na;b;1;1\n" + b"a;b;1;1\n" * 100_000
    importer = AsyncMock(return_value=ImportResponse(
        import_id=uuid4(), kind="payments", filename="payments.csv", row_count=100_001,
        encoding="utf-8-sig", delimiter=";", reused=False, counts=ImportCounts(payments=100_001),
    ))
    monkeypatch.setattr("app.api.imports.import_csv", importer)
    app = create_app(settings)
    app.dependency_overrides[get_session] = lambda: AsyncMock()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=uuid4())
    with TestClient(app) as client:
        result = client.post("/imports/payments", files={"file": ("payments.csv", content)})
    assert result.status_code == 200
    assert result.json()["row_count"] == 100_001
    assert len(importer.await_args.args[2].rows) == 100_001
    assert importer.await_args.args[2].rows[-1]["source_row_number"] == 100_002


def test_person_updates_preserve_all_links_and_report_conflicts(monkeypatch):
    content = b"SYNTHETIC_F;SYNTHETIC_U;CITY_NAME;RC_NAME\nf;a;One;R\nf;b;Two;S\nf;a;;\n"
    parsed = parse_csv(content, "persons", "utf-8-sig", None, 10)
    assert parsed.person_attribute_conflicts == 2
    upsert = AsyncMock()
    monkeypatch.setattr("app.repositories.csv_import.upsert_rows", upsert)
    monkeypatch.setattr("app.repositories.csv_import.ensure_companies", AsyncMock(return_value=0))
    counts = asyncio.run(save_persons(AsyncMock(), parsed.rows, uuid4()))
    assert counts["persons"] == 1
    assert counts["links"] == 2
    person = upsert.await_args_list[0].args[2][0]
    assert person["city"] is None
    assert person["region"] is None
    assert {row["company_id"] for row in upsert.await_args_list[1].args[2]} == {"a", "b"}
    record = SimpleNamespace(
        id=uuid4(), kind="persons", filename="persons.csv", row_count=3,
        encoding="utf-8-sig", delimiter=";",
        counts={**counts, "person_attribute_conflicts": parsed.person_attribute_conflicts},
    )
    for reused in (False, True):
        result = response(record, reused=reused)
        assert result.counts.person_attribute_conflicts == 2
        assert result.warnings


def test_identical_person_attributes_do_not_report_conflicts():
    content = b"SYNTHETIC_F;SYNTHETIC_U;CITY_NAME;RC_NAME\nf;a;One;R\nf;b;One;R\n"
    assert parse_csv(content, "persons", "utf-8-sig", None, 10).person_attribute_conflicts == 0
