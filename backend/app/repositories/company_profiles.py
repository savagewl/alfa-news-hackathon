"""Database preparation and persistent cache operations for company profiles."""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import and_, case, func, literal, not_, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.models.company import Company, CompanyPayment
from app.models.company_profile import CompanyProfileCache
from app.services.llm.company_profile import build_company_profile_text
from app.services.llm.config import ServicesSettings
from app.services.llm.schemas import CompanyFields, CompanyProfile, CompanyProfileText, Counterparty

PROFILE_INPUT_VERSION = 1
MAX_PROFILE_COUNTERPARTIES = 10


@dataclass(frozen=True)
class CompanyProfileInput:
    fingerprint: str
    fields: CompanyFields
    text: CompanyProfileText
    embedding_model_version: str


def configured_embedding_model_version(settings: ServicesSettings) -> str:
    """Return the configured version without constructing an embeddings HTTP client."""
    return f"{settings.embed_model}:{settings.embed_dim}d"


def company_fields(company: Company, counterparties: list[Counterparty]) -> CompanyFields:
    products = [
        code
        for code, enabled in (
            ("rko", company.has_rko),
            ("gar", company.has_guarantees),
            ("loa", company.has_credit),
            ("lea", company.has_leasing),
            ("dep", company.has_deposit),
            ("zpp", company.has_payroll),
        )
        if enabled is True
    ]
    revenue = float(company.annual_revenue) if company.annual_revenue is not None else None
    return CompanyFields(
        synthetic_u=company.id,
        org_type=company.org_type,
        okfs_name=company.ownership_form,
        company_age_years=company.age_category,
        eco_stage=company.development_stage,
        okved_code=company.okved_code,
        okved_name=company.okved_name,
        # CSV NEGATIVEMARK is stored in this legacy-named boolean column.
        negative_mark=(
            None if company.key_owner_flag is None else str(int(company.key_owner_flag))
        ),
        workers_fns=company.employees_fns_category,
        workers_spark=company.employees_spark_category,
        client_type=company.client_type,
        products=products,
        fts_flag=company.has_fts_data is True,
        fts_country_name=company.fts_country_name,
        factoring=company.has_factoring is True,
        revenue=revenue,
        group_id=company.group_id,
        city=company.city,
        region=company.region,
        counterparties=counterparties,
    )


async def load_counterparties(
    session: AsyncSession, company_id: str, *, limit: int = MAX_PROFILE_COUNTERPARTIES
) -> list[Counterparty]:
    other = aliased(Company)
    incoming = (
        select(
            other.id.label("company_id"),
            other.name.label("name"),
            literal("incoming").label("direction"),
            func.sum(CompanyPayment.payment_amount).label("pay_amt"),
            func.sum(CompanyPayment.payment_count).label("pay_cnt"),
        )
        .join(other, other.id == CompanyPayment.payer_company_id)
        .where(CompanyPayment.beneficiary_company_id == company_id)
        .group_by(other.id, other.name)
    )
    outgoing = (
        select(
            other.id.label("company_id"),
            other.name.label("name"),
            literal("outgoing").label("direction"),
            func.sum(CompanyPayment.payment_amount).label("pay_amt"),
            func.sum(CompanyPayment.payment_count).label("pay_cnt"),
        )
        .join(other, other.id == CompanyPayment.beneficiary_company_id)
        .where(CompanyPayment.payer_company_id == company_id)
        .group_by(other.id, other.name)
    )
    combined = incoming.union_all(outgoing).subquery()
    rows = await session.execute(
        select(combined)
        .order_by(combined.c.pay_amt.desc(), combined.c.company_id, combined.c.direction)
        .limit(limit)
    )
    return [
        Counterparty(
            name=row.name or row.company_id,
            direction=row.direction,
            pay_amt=float(row.pay_amt or Decimal(0)),
            pay_cnt=int(row.pay_cnt or 0),
        )
        for row in rows
    ]


async def prepare_company_profile_input(
    session: AsyncSession, company: Company, settings: ServicesSettings
) -> CompanyProfileInput:
    counterparties = await load_counterparties(session, company.id)
    return build_company_profile_input(company, counterparties, settings)


def build_company_profile_input(
    company: Company,
    counterparties: list[Counterparty],
    settings: ServicesSettings,
) -> CompanyProfileInput:
    fields = company_fields(company, counterparties)
    text = build_company_profile_text(fields)
    model_version = configured_embedding_model_version(settings)
    canonical = {
        "version": PROFILE_INPUT_VERSION,
        "embedding_model_version": model_version,
        "fields": fields.model_dump(mode="json"),
        "text": text.model_dump(mode="json"),
    }
    fingerprint = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return CompanyProfileInput(fingerprint, fields, text, model_version)


async def load_current_company_profile(
    session: AsyncSession, prepared: CompanyProfileInput
) -> CompanyProfile | None:
    cache = await session.scalar(
        select(CompanyProfileCache).where(
            CompanyProfileCache.company_id == prepared.fields.synthetic_u,
            CompanyProfileCache.status == "ready",
            CompanyProfileCache.input_fingerprint == prepared.fingerprint,
            CompanyProfileCache.embedding_model_version == prepared.embedding_model_version,
        )
    )
    if cache is None:
        return None
    vectors = cache.vectors or {}
    if set(prepared.text.aspects).difference(vectors) or "main" not in vectors:
        return None
    return CompanyProfile(
        text=CompanyProfileText.model_validate(cache.profile_text),
        vectors=vectors,
        embedding_model_version=cache.embedding_model_version,
    )


async def claim_company_profile(
    session: AsyncSession,
    prepared: CompanyProfileInput,
    *,
    lease_seconds: int,
) -> UUID | None:
    now, token = datetime.now(UTC), uuid4()
    values = {
        "company_id": prepared.fields.synthetic_u,
        "input_fingerprint": prepared.fingerprint,
        "profile_text": prepared.text.model_dump(mode="json"),
        "fields": prepared.fields.model_dump(mode="json"),
        "vectors": {},
        "embedding_model_version": prepared.embedding_model_version,
        "status": "processing",
        "attempts": 0,
        "next_attempt_at": None,
        "last_error": None,
        "lease_token": token,
        "lease_until": now + timedelta(seconds=lease_seconds),
        "updated_at": now,
    }
    different_input = or_(
        CompanyProfileCache.input_fingerprint != prepared.fingerprint,
        CompanyProfileCache.embedding_model_version != prepared.embedding_model_version,
    )
    retry_due = (CompanyProfileCache.status == "retry") & or_(
        CompanyProfileCache.next_attempt_at.is_(None), CompanyProfileCache.next_attempt_at <= now
    )
    lease_expired = (CompanyProfileCache.status == "processing") & or_(
        CompanyProfileCache.lease_until.is_(None), CompanyProfileCache.lease_until <= now
    )
    active_lease = and_(
        CompanyProfileCache.status == "processing",
        CompanyProfileCache.lease_until.is_not(None),
        CompanyProfileCache.lease_until > now,
    )
    statement = insert(CompanyProfileCache).values(**values)
    updates = {key: getattr(statement.excluded, key) for key in values if key != "company_id"}
    updates["attempts"] = case(
        (different_input, 0), else_=CompanyProfileCache.attempts
    )
    claimed = await session.scalar(
        statement.on_conflict_do_update(
            index_elements=["company_id"],
            set_=updates,
            where=or_(
                and_(different_input, not_(active_lease)),
                retry_due,
                lease_expired,
                CompanyProfileCache.status == "pending",
            ),
        ).returning(CompanyProfileCache.company_id)
    )
    await session.commit()
    return token if claimed is not None else None


async def complete_company_profile(
    session: AsyncSession,
    prepared: CompanyProfileInput,
    token: UUID,
    vectors: dict[str, list[float]],
) -> bool:
    saved = await session.scalar(
        update(CompanyProfileCache)
        .where(
            CompanyProfileCache.company_id == prepared.fields.synthetic_u,
            CompanyProfileCache.input_fingerprint == prepared.fingerprint,
            CompanyProfileCache.lease_token == token,
            CompanyProfileCache.status == "processing",
        )
        .values(
            vectors=vectors,
            status="ready",
            attempts=0,
            next_attempt_at=None,
            last_error=None,
            lease_token=None,
            lease_until=None,
            updated_at=datetime.now(UTC),
        )
        .returning(CompanyProfileCache.company_id)
    )
    await session.commit()
    return saved is not None


async def fail_company_profile(
    session: AsyncSession,
    company_id: str,
    token: UUID,
    *,
    retry_seconds: int,
    error: str,
) -> None:
    now = datetime.now(UTC)
    await session.execute(
        update(CompanyProfileCache)
        .where(
            CompanyProfileCache.company_id == company_id,
            CompanyProfileCache.lease_token == token,
            CompanyProfileCache.status == "processing",
        )
        .values(
            status="retry",
            attempts=CompanyProfileCache.attempts + 1,
            next_attempt_at=now + timedelta(seconds=retry_seconds),
            last_error=error[:1000],
            lease_token=None,
            lease_until=None,
            updated_at=now,
        )
    )
    await session.commit()
