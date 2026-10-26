import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.config import Settings
from app.db.session import get_session
from app.main import create_app
from app.repositories.csv_import import save_persons
from app.repositories.import_access import accessible_person_links, authorize_import
from app.schemas.csv_import import ImportCounts, ImportResponse
from app.services.csv_import import import_csv
from app.services.csv_parser import parse_csv


def test_partial_person_import_retries_with_current_access(monkeypatch):
    async def check():
        content = b"SYNTHETIC_F;SYNTHETIC_U;CITY_NAME;RC_NAME\nf;a;One;R\nf;b;Two;S\ng;c;Three;T\n"
        parsed = parse_csv(content, "persons", "utf-8-sig", None, 100)
        session = AsyncMock()
        session.scalars.return_value = ["a"]
        records = {}
        session.add = Mock(
            side_effect=lambda record: records.setdefault(record.content_hash, record)
        )
        authorize = AsyncMock(return_value=set())
        monkeypatch.setattr("app.services.csv_import.lock_imports", AsyncMock())
        monkeypatch.setattr("app.services.csv_import.authorize_import", authorize)
        monkeypatch.setattr("app.services.csv_import.join_new_companies", AsyncMock())

        async def find(session, kind, content_hash):
            return records.get(content_hash)

        async def save(session, rows, import_id, **kwargs):
            return {"persons": len({row["id"] for row in rows}), "links": len(rows)}

        saver = AsyncMock(side_effect=save)
        monkeypatch.setattr("app.services.csv_import.find_import", find)
        monkeypatch.setattr("app.services.csv_import.save_persons", saver)
        user_id = uuid4()

        async def upload(user=user_id):
            return await import_csv(
                session, "persons", parsed, content, "persons.csv", "utf-8-sig", user,
                skip_inaccessible_person_links=True,
            )

        result = await upload()
        assert result.row_count == 3
        assert result.counts.links == 1
        assert result.counts.skipped_inaccessible_rows == 2
        assert result.counts.person_attribute_conflicts == 0
        assert result.warnings
        assert saver.await_args.args[1] == parsed.rows[:1]
        assert authorize.await_args.args[2] == parsed.rows[:1]
        repeat = await upload()
        assert repeat.reused
        assert repeat.import_id == result.import_id
        assert repeat.counts == result.counts
        assert repeat.warnings == result.warnings
        saver.assert_awaited_once()
        # Same CSV and same accessible companies for a second user must not reuse the first log.
        second = await upload(uuid4())
        assert not second.reused
        assert second.import_id != result.import_id
        # Newly accessible companies must be saved instead of returning the old partial result.
        session.scalars.return_value = ["a", "b"]
        expanded = await upload()
        assert not expanded.reused
        assert expanded.counts.links == 2
        assert expanded.counts.skipped_inaccessible_rows == 1
        assert expanded.counts.person_attribute_conflicts == 1
        assert saver.await_args.args[1] == parsed.rows[:2]
        # No access: report all skips, save no persons or links and grant no memberships.
        session.scalars.return_value = []
        empty = await upload()
        assert empty.counts.links == 0
        assert empty.counts.skipped_inaccessible_rows == 3
        assert saver.await_args.args[1] == []
        # The new default imports all rows even with no company memberships.
        full = await import_csv(
            session, "persons", parsed, content, "persons.csv", "utf-8-sig", user_id,
        )
        assert not full.reused
        assert full.counts.links == 3
        assert full.counts.skipped_inaccessible_rows == 0
        assert full.counts.person_attribute_conflicts == 1
        assert saver.await_args.args[1] == parsed.rows
        assert saver.await_args.kwargs["user_id"] == user_id
        repeated = await import_csv(
            session, "persons", parsed, content, "persons.csv", "utf-8-sig", user_id,
        )
        assert repeated.reused
        assert repeated.import_id == full.import_id

    asyncio.run(check())


def test_filter_only_keeps_owned_company_links():
    async def check():
        session = AsyncMock()
        session.scalars.return_value = ["own"]
        rows = [{"company_id": "own"}, {"company_id": "foreign"}, {"company_id": "unknown"}]
        assert await accessible_person_links(session, rows, uuid4()) == rows[:1]

    asyncio.run(check())


def test_shared_person_attributes_are_preserved_but_all_links_are_saved(monkeypatch):
    async def check():
        session = AsyncMock()
        session.scalars.return_value = [SimpleNamespace(id="f", city="Old", region="R")]
        monkeypatch.setattr(
            "app.repositories.csv_import.ensure_companies", AsyncMock(return_value=1)
        )
        upsert = AsyncMock()
        monkeypatch.setattr("app.repositories.csv_import.upsert_rows", upsert)
        rows = [
            {"company_id": company, "id": "f", "city": "Changed", "region": "R"}
            for company in ["own", "foreign", "unknown"]
        ]
        user_id = uuid4()
        assert await authorize_import(session, "persons", rows, user_id) == set()
        result = await save_persons(session, rows, uuid4(), user_id=user_id)
        assert result["links"] == 3
        assert result["preserved_person_attributes"] == 1
        assert upsert.await_args_list[0].args[2] == []
        links = upsert.await_args_list[1].args[2]
        assert {link["company_id"] for link in links} == {"own", "foreign", "unknown"}

    asyncio.run(check())


@pytest.mark.parametrize("query,skip", [("", False), ("?skip_inaccessible=true", True)])
def test_person_endpoint_imports_all_links_with_optional_filtering(monkeypatch, query, skip):
    importer = AsyncMock(return_value=ImportResponse(
        import_id=uuid4(), kind="persons", filename="persons.csv", row_count=1,
        encoding="utf-8-sig", delimiter=",", reused=False, counts=ImportCounts(links=1),
    ))
    monkeypatch.setattr("app.api.imports.import_csv", importer)
    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_session] = lambda: AsyncMock()
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=uuid4())
    with TestClient(app) as client:
        response = client.post(
            "/imports/persons" + query,
            files={"file": (
                "persons.csv", b"SYNTHETIC_F,SYNTHETIC_U,CITY_NAME,RC_NAME\nf,a,M,R\n"
            )},
        )
    assert response.status_code == 200
    assert importer.await_args.kwargs["skip_inaccessible_person_links"] is skip
