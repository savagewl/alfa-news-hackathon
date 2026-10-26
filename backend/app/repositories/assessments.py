"""Queue and cache of short per-company news assessments (table news_assessments)."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import and_, case, null, or_, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.news_assessment import NewsAssessment
from app.repositories.interpretations import interpretation_evidence_is_current

QUEUED_STATUSES = ("pending", "retry", "processing")
# После окончательного сбоя новость не ставится в очередь снова, пока не пройдёт эта пауза.
FAILED_COOLDOWN = timedelta(minutes=10)


@dataclass(frozen=True)
class AssessmentClaim:
    company_id: str
    news_id: UUID
    lease_token: UUID
    attempt: int


async def enqueue_assessments(
    session: AsyncSession, company_id: str, fingerprints: dict[UUID, str]
) -> set[UUID]:
    """Queue missing or outdated assessments. Returns ids that were inserted or reset."""

    if not fingerprints:
        return set()
    now = datetime.now(UTC)
    statement = insert(NewsAssessment).values(
        [
            {
                "company_id": company_id,
                "news_id": news_id,
                "input_fingerprint": fingerprint,
                "status": "pending",
                "attempts": 0,
                "updated_at": now,
            }
            for news_id, fingerprint in fingerprints.items()
        ]
    )
    lease_free = or_(NewsAssessment.status != "processing", NewsAssessment.lease_until < now)
    changed = await session.scalars(
        statement.on_conflict_do_update(
            index_elements=[NewsAssessment.company_id, NewsAssessment.news_id],
            set_={
                "input_fingerprint": statement.excluded.input_fingerprint,
                "status": "pending",
                "attempts": 0,
                "next_attempt_at": null(),
                "last_error": null(),
                "result": null(),
                "expires_at": null(),
                "lease_token": null(),
                "lease_until": null(),
                "updated_at": now,
            },
            where=or_(
                and_(
                    NewsAssessment.input_fingerprint != statement.excluded.input_fingerprint,
                    lease_free,
                ),
                and_(NewsAssessment.status == "ready", NewsAssessment.expires_at < now),
                and_(
                    NewsAssessment.status == "failed",
                    NewsAssessment.updated_at < now - FAILED_COOLDOWN,
                ),
            ),
        ).returning(NewsAssessment.news_id)
    )
    queued = set(changed)
    await session.commit()
    return queued


async def claim_assessments(
    session: AsyncSession, *, limit: int, lease_seconds: int
) -> list[AssessmentClaim]:
    now, token = datetime.now(UTC), uuid4()
    due = or_(
        NewsAssessment.status == "pending",
        and_(
            NewsAssessment.status == "retry",
            or_(NewsAssessment.next_attempt_at.is_(None), NewsAssessment.next_attempt_at <= now),
        ),
        and_(
            NewsAssessment.status == "processing",
            or_(NewsAssessment.lease_until.is_(None), NewsAssessment.lease_until <= now),
        ),
    )
    picked = (
        select(NewsAssessment.company_id, NewsAssessment.news_id)
        .where(due)
        .order_by(NewsAssessment.updated_at.desc())
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    rows = await session.execute(
        update(NewsAssessment)
        .where(tuple_(NewsAssessment.company_id, NewsAssessment.news_id).in_(picked))
        .values(
            status="processing",
            attempts=NewsAssessment.attempts + 1,
            lease_token=token,
            lease_until=now + timedelta(seconds=lease_seconds),
            updated_at=now,
        )
        .returning(NewsAssessment.company_id, NewsAssessment.news_id, NewsAssessment.attempts)
    )
    claims = [
        AssessmentClaim(company_id, news_id, token, attempts)
        for company_id, news_id, attempts in rows.all()
    ]
    await session.commit()
    return claims


async def claim_assessment(
    session: AsyncSession,
    company_id: str,
    news_id: UUID,
    fingerprint: str,
    *,
    lease_seconds: int,
) -> AssessmentClaim | None:
    """Take the generation lease for one (company, news) pair; None if someone else holds it."""

    now, token = datetime.now(UTC), uuid4()
    statement = insert(NewsAssessment).values(
        company_id=company_id,
        news_id=news_id,
        input_fingerprint=fingerprint,
        status="processing",
        attempts=1,
        lease_token=token,
        lease_until=now + timedelta(seconds=lease_seconds),
        updated_at=now,
    )
    claimed = await session.scalar(
        statement.on_conflict_do_update(
            index_elements=[NewsAssessment.company_id, NewsAssessment.news_id],
            set_={
                "input_fingerprint": fingerprint,
                "status": "processing",
                "attempts": 1,
                "next_attempt_at": null(),
                "last_error": null(),
                "result": null(),
                "expires_at": null(),
                "lease_token": token,
                "lease_until": now + timedelta(seconds=lease_seconds),
                "updated_at": now,
            },
            where=and_(
                or_(
                    NewsAssessment.status != "processing",
                    NewsAssessment.lease_until.is_(None),
                    NewsAssessment.lease_until <= now,
                ),
                or_(
                    NewsAssessment.status != "ready",
                    NewsAssessment.input_fingerprint != fingerprint,
                    NewsAssessment.expires_at.is_(None),
                    NewsAssessment.expires_at <= now,
                ),
            ),
        ).returning(NewsAssessment.news_id)
    )
    await session.commit()
    return AssessmentClaim(company_id, news_id, token, 1) if claimed is not None else None


async def complete_assessment(
    session: AsyncSession,
    claim: AssessmentClaim,
    *,
    fingerprint: str,
    result: dict,
    cache_hours: int,
) -> bool:
    now = datetime.now(UTC)
    saved = await session.scalar(
        update(NewsAssessment)
        .where(
            NewsAssessment.company_id == claim.company_id,
            NewsAssessment.news_id == claim.news_id,
            NewsAssessment.lease_token == claim.lease_token,
            NewsAssessment.status == "processing",
        )
        .values(
            input_fingerprint=fingerprint,
            status="ready",
            result=result,
            attempts=0,
            next_attempt_at=None,
            last_error=None,
            lease_token=None,
            lease_until=None,
            expires_at=now + timedelta(hours=cache_hours),
            updated_at=now,
        )
        .returning(NewsAssessment.news_id)
    )
    await session.commit()
    return saved is not None


async def fail_assessment(
    session: AsyncSession,
    claim: AssessmentClaim,
    *,
    error: str,
    retry_seconds: int,
    max_attempts: int,
) -> None:
    now = datetime.now(UTC)
    exhausted = NewsAssessment.attempts >= max_attempts
    await session.execute(
        update(NewsAssessment)
        .where(
            NewsAssessment.company_id == claim.company_id,
            NewsAssessment.news_id == claim.news_id,
            NewsAssessment.lease_token == claim.lease_token,
            NewsAssessment.status == "processing",
        )
        .values(
            status=case((exhausted, "failed"), else_="retry"),
            next_attempt_at=case(
                (exhausted, null()), else_=now + timedelta(seconds=retry_seconds)
            ),
            last_error=error[:1000],
            lease_token=None,
            lease_until=None,
            updated_at=now,
        )
    )
    await session.commit()


async def load_assessment_states(
    session: AsyncSession,
    company_id: str,
    fingerprints: dict[UUID, str],
    *,
    trusted_bank_document_ids: list[UUID],
) -> tuple[dict[UUID, dict], set[UUID]]:
    """Ready assessments with current evidence, and ids that are queued or in progress."""

    if not fingerprints:
        return {}, set()
    now = datetime.now(UTC)
    rows = await session.scalars(
        select(NewsAssessment).where(
            NewsAssessment.company_id == company_id,
            NewsAssessment.news_id.in_(fingerprints),
        )
    )
    ready: dict[UUID, dict] = {}
    queued: set[UUID] = set()
    for row in rows:
        if row.input_fingerprint != fingerprints.get(row.news_id):
            continue
        if row.status in QUEUED_STATUSES:
            queued.add(row.news_id)
            continue
        payload = row.result
        if (
            row.status != "ready"
            or not isinstance(payload, dict)
            or row.expires_at is None
            or row.expires_at <= now
        ):
            continue
        if not await interpretation_evidence_is_current(
            session, payload, trusted_bank_document_ids=trusted_bank_document_ids
        ):
            # Make this row reclaimable, including when called outside the feed queue.
            await session.execute(
                update(NewsAssessment)
                .where(
                    NewsAssessment.company_id == company_id,
                    NewsAssessment.news_id == row.news_id,
                    NewsAssessment.status == "ready",
                    NewsAssessment.input_fingerprint == row.input_fingerprint,
                    NewsAssessment.updated_at == row.updated_at,
                )
                .values(expires_at=now)
            )
            await session.commit()
            continue
        ready[row.news_id] = payload
    return ready, queued
