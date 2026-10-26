from collections.abc import Iterable

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.company import Company, CompanyPayment, CompanyPersonLink, CsvImport, Person

BATCH_SIZE = 500


async def lock_imports(session: AsyncSession) -> None:
    # Serialize import writes, including cross-dataset placeholder creation.
    # The transaction lock is also released on failure/disconnect.
    await session.execute(text("SELECT pg_advisory_xact_lock(72841931)"))


async def find_import(session: AsyncSession, kind: str, content_hash: str) -> CsvImport | None:
    return await session.scalar(
        select(CsvImport).where(CsvImport.kind == kind, CsvImport.content_hash == content_hash)
    )


async def upsert_rows(
    session: AsyncSession, model: type, rows: list[dict], keys: list[str]
) -> None:
    for offset in range(0, len(rows), BATCH_SIZE):
        batch = rows[offset : offset + BATCH_SIZE]
        statement = insert(model).values(batch)
        changes = {key: statement.excluded[key] for key in batch[0] if key not in keys}
        if model is Company:
            changes["updated_at"] = func.now()
        await session.execute(statement.on_conflict_do_update(index_elements=keys, set_=changes))


async def ensure_companies(session: AsyncSession, ids: Iterable[str]) -> int:
    rows = [{"id": key, "is_placeholder": True} for key in sorted(set(ids))]
    created = 0
    for offset in range(0, len(rows), BATCH_SIZE):
        statement = insert(Company).values(rows[offset : offset + BATCH_SIZE])
        result = await session.execute(statement.on_conflict_do_nothing(index_elements=["id"]))
        created += result.rowcount
    return created


async def save_companies(session: AsyncSession, rows: list[dict], import_id) -> dict:
    # Preserve file order: a later row replaces the whole earlier row for the same ID.
    unique = {
        row["id"]: {**row, "is_placeholder": False, "last_import_id": import_id} for row in rows
    }
    await upsert_rows(session, Company, list(unique.values()), ["id"])
    return {"companies": len(unique)}


async def save_persons(session: AsyncSession, rows: list[dict], import_id) -> dict:
    placeholders = await ensure_companies(session, (row["company_id"] for row in rows))
    persons = {}
    links = {}
    for row in rows:
        persons[row["id"]] = {
            "id": row["id"],
            "city": row["city"],
            "region": row["region"],
            "last_import_id": import_id,
        }
        links[row["company_id"], row["id"]] = {
            "company_id": row["company_id"],
            "person_id": row["id"],
            "last_import_id": import_id,
        }
    await upsert_rows(session, Person, list(persons.values()), ["id"])
    await upsert_rows(session, CompanyPersonLink, list(links.values()), ["company_id", "person_id"])
    return {"persons": len(persons), "links": len(links), "placeholder_companies": placeholders}


async def save_payments(session: AsyncSession, rows: list[dict], import_id) -> dict:
    ids = {row[key] for row in rows for key in ("payer_company_id", "beneficiary_company_id")}
    placeholders = await ensure_companies(session, ids)
    for offset in range(0, len(rows), BATCH_SIZE):
        await session.execute(
            insert(CompanyPayment).values(
                [{**row, "import_id": import_id} for row in rows[offset : offset + BATCH_SIZE]]
            )
        )
    return {"payments": len(rows), "placeholder_companies": placeholders}
