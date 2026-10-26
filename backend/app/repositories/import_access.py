from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Company, CompanyMembership
from app.repositories.csv_import import BATCH_SIZE
from app.schemas.csv_import import ImportKind


class ImportAccessError(PermissionError):
    pass


async def accessible_person_links(
    session: AsyncSession, rows: list[dict], user_id: UUID
) -> list[dict]:
    owned = set(await session.scalars(
        select(CompanyMembership.company_id).where(CompanyMembership.user_id == user_id)
    ))
    return [row for row in rows if row["company_id"] in owned]


async def authorize_import(
    session: AsyncSession,
    kind: ImportKind,
    rows: list[dict],
    user_id: UUID,
) -> set[str]:
    """Called under the import transaction lock; returns new companies to join."""
    if kind == "persons":
        # A relationship contributes to analysis; it does not grant company membership.
        # save_persons preserves attributes already used by inaccessible companies.
        return set()
    owned = set(
        await session.scalars(
            select(CompanyMembership.company_id).where(
                CompanyMembership.user_id == user_id,
            )
        )
    )
    if kind == "companies":
        ids = sorted({row["id"] for row in rows})
        existing = set()
        for offset in range(0, len(ids), BATCH_SIZE):
            existing.update(
                await session.scalars(
                    select(Company.id).where(
                        Company.id.in_(ids[offset : offset + BATCH_SIZE]),
                    )
                )
            )
        if existing - owned:
            raise ImportAccessError("CSV contains companies outside your access")
        return set(ids) - existing
    if kind == "payments":
        if any(
            row["payer_company_id"] not in owned and row["beneficiary_company_id"] not in owned
            for row in rows
        ):
            raise ImportAccessError("Every payment must involve a company you belong to")
        return set()
    return set()


async def join_new_companies(
    session: AsyncSession,
    company_ids: set[str],
    user_id: UUID,
) -> None:
    ids = sorted(company_ids)
    for offset in range(0, len(ids), BATCH_SIZE):
        await session.execute(
            insert(CompanyMembership)
            .values(
                [
                    {"company_id": company_id, "user_id": user_id, "added_by_user_id": user_id}
                    for company_id in ids[offset : offset + BATCH_SIZE]
                ]
            )
            .on_conflict_do_nothing(index_elements=["company_id", "user_id"])
        )
