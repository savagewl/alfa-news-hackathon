from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analytics import AnalyticsChunk, AnalyticsDocument
from app.models.interpretation import NewsInterpretation
from app.services.llm.text import is_empty_assessment, shorten_text


async def acquire_interpretation(
    session: AsyncSession,
    *,
    company_id: str,
    news_id: UUID,
    request_fingerprint: str,
    company_input_fingerprint: str,
    news_input_fingerprint: str,
    model_prompt_version: str,
    timeout_seconds: int,
    cache_hours: int,
) -> tuple[NewsInterpretation, UUID | None]:
    """Acquire one generation lease per stable input request.

    ``cache_key`` starts as the request fingerprint and is replaced with the final
    context key after verifier-directed retrieval.  ``request_fingerprint`` remains
    stable, so a later request can still find and reuse the completed row.
    """

    now, token = datetime.now(UTC), uuid4()
    values = dict(
        company_id=company_id,
        news_id=news_id,
        cache_key=request_fingerprint,
        request_fingerprint=request_fingerprint,
        company_input_fingerprint=company_input_fingerprint,
        news_input_fingerprint=news_input_fingerprint,
        retrieval_fingerprint=None,
        model_prompt_version=model_prompt_version,
        status="generating",
        lease_token=token,
        lease_until=now + timedelta(seconds=timeout_seconds + 30),
        expires_at=now + timedelta(hours=cache_hours),
        updated_at=now,
        result=None,
        error=None,
    )
    await session.execute(
        insert(NewsInterpretation)
        .values(id=uuid4(), **values)
        .on_conflict_do_update(
            index_elements=["request_fingerprint"],
            set_=values,
            where=or_(
                NewsInterpretation.status == "failed",
                (NewsInterpretation.status == "generating")
                & (NewsInterpretation.lease_until < now),
                (NewsInterpretation.status == "completed")
                & (NewsInterpretation.expires_at < now),
            ),
        )
    )
    record = await session.scalar(
        select(NewsInterpretation)
        .where(NewsInterpretation.request_fingerprint == request_fingerprint)
        .execution_options(populate_existing=True)
    )
    await session.commit()
    return record, token if record and record.lease_token == token else None


async def complete_interpretation(
    session: AsyncSession,
    record_id: UUID,
    token: UUID,
    *,
    result: dict | None,
    final_cache_key: str | None = None,
    retrieval_fingerprint: str | None = None,
    error: str | None = None,
) -> bool:
    values = dict(
        status="failed" if error else "completed",
        result=result,
        error=error,
        lease_token=None,
        lease_until=None,
        updated_at=datetime.now(UTC),
    )
    if final_cache_key is not None:
        values["cache_key"] = final_cache_key
    if retrieval_fingerprint is not None:
        values["retrieval_fingerprint"] = retrieval_fingerprint
    saved = await session.scalar(
        update(NewsInterpretation)
        .where(
            NewsInterpretation.id == record_id,
            NewsInterpretation.lease_token == token,
            NewsInterpretation.status == "generating",
        )
        .values(**values)
        .returning(NewsInterpretation.id)
    )
    await session.commit()
    return saved is not None


def _brief(value, *, sentences: int, chars: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return shorten_text(value, sentences=sentences, chars=chars)


def _brief_list(values, *, limit: int = 2) -> list[str] | None:
    """Для ленты: не больше двух пунктов по одному короткому предложению; None, если пусто."""

    items = (
        _brief(value, sentences=1, chars=200)
        for value in (values or [])[:limit]
        if not is_empty_assessment(value)
    )
    return [item for item in items if item] or None


async def load_ready_assessments(
    session: AsyncSession,
    *,
    company_id: str,
    news_fingerprints: dict[UUID, str],
    company_input_fingerprint: str,
    model_prompt_version: str,
    trusted_bank_document_ids: list[UUID],
) -> dict[UUID, dict]:
    """Load compact, currently valid assessments using DB reads only."""

    if not news_fingerprints:
        return {}
    now = datetime.now(UTC)
    rows = await session.scalars(
        select(NewsInterpretation)
        .where(
            NewsInterpretation.company_id == company_id,
            NewsInterpretation.news_id.in_(news_fingerprints),
            NewsInterpretation.status == "completed",
            NewsInterpretation.expires_at > now,
            NewsInterpretation.company_input_fingerprint == company_input_fingerprint,
            NewsInterpretation.model_prompt_version == model_prompt_version,
        )
        .order_by(NewsInterpretation.updated_at.desc())
    )
    assessments: dict[UUID, dict] = {}
    for row in rows:
        if row.news_id in assessments:
            continue
        if row.news_input_fingerprint != news_fingerprints.get(row.news_id):
            continue
        payload = row.result or {}
        interpretation = payload.get("interpretation")
        verification = payload.get("verification") or {}
        if (
            payload.get("status") != "ready"
            or verification.get("status") != "APPROVED"
            or not isinstance(interpretation, dict)
        ):
            continue
        if not await interpretation_evidence_is_current(
            session,
            payload,
            trusted_bank_document_ids=trusted_bank_document_ids,
        ):
            continue
        assessments[row.news_id] = {
            "assessment_status": "ready",
            "brief_impact": _brief(interpretation.get("ai_interpretation"), sentences=2, chars=280),
            "risks": _brief_list(interpretation.get("risks")),
            "opportunities": _brief_list(interpretation.get("opportunities")),
            "bank_analytics": interpretation.get("bank_position"),
            "bank_sources": [
                source
                for source in payload.get("sources") or []
                if str(source.get("evidence_ref", "")).startswith("bank:")
            ]
            if interpretation.get("bank_position")
            else [],
            "generated_at": payload.get("generated_at"),
        }
    return assessments


async def get_ready_interpretation(
    session: AsyncSession,
    *,
    request_fingerprint: str,
    trusted_bank_document_ids: list[UUID],
) -> NewsInterpretation | None:
    """Return only an unexpired, verifier-approved current interpretation."""

    row = await session.scalar(
        select(NewsInterpretation)
        .where(
            NewsInterpretation.request_fingerprint == request_fingerprint,
            NewsInterpretation.status == "completed",
            NewsInterpretation.expires_at > datetime.now(UTC),
        )
        .order_by(NewsInterpretation.updated_at.desc())
        .limit(1)
    )
    payload = row.result if row else None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "ready" or not payload.get("interpretation"):
        return None
    if (payload.get("verification") or {}).get("status") != "APPROVED":
        return None
    if not await interpretation_evidence_is_current(
        session,
        payload,
        trusted_bank_document_ids=trusted_bank_document_ids,
    ):
        return None
    return row


async def find_current_interpretation(
    session: AsyncSession,
    *,
    company_id: str,
    news_id: UUID,
    company_input_fingerprint: str,
    news_input_fingerprint: str,
    model_prompt_version: str,
    trusted_bank_document_ids: list[UUID],
) -> NewsInterpretation | None:
    """Find approved evidence by stable inputs, including rows with legacy cache keys."""
    rows = await session.scalars(
        select(NewsInterpretation)
        .where(
            NewsInterpretation.company_id == company_id,
            NewsInterpretation.news_id == news_id,
            NewsInterpretation.company_input_fingerprint == company_input_fingerprint,
            NewsInterpretation.news_input_fingerprint == news_input_fingerprint,
            NewsInterpretation.model_prompt_version == model_prompt_version,
            NewsInterpretation.status == "completed",
            NewsInterpretation.expires_at > datetime.now(UTC),
        )
        .order_by(NewsInterpretation.updated_at.desc())
    )
    for row in rows:
        payload = row.result or {}
        if (
            payload.get("status") == "ready"
            and payload.get("interpretation")
            and (payload.get("verification") or {}).get("status") == "APPROVED"
            and await interpretation_evidence_is_current(
                session, payload, trusted_bank_document_ids=trusted_bank_document_ids
            )
        ):
            return row
    return None


async def interpretation_evidence_is_current(
    session: AsyncSession,
    payload: dict,
    *,
    trusted_bank_document_ids: list[UUID],
) -> bool:
    """Validate stored bank passages with one bounded DB query, without retrieval/AI."""

    stored = payload.get("bank_passages") or []
    if not stored:
        return True
    trusted = set(trusted_bank_document_ids)
    expected: dict[tuple[UUID, str], dict] = {}
    try:
        for passage in stored:
            key = (UUID(str(passage["document_id"])), str(passage["chunk_id"]))
            if key[0] not in trusted:
                return False
            expected[key] = passage
    except (KeyError, TypeError, ValueError):
        return False
    rows = await session.execute(
        select(AnalyticsChunk, AnalyticsDocument)
        .join(AnalyticsDocument)
        .where(
            AnalyticsChunk.document_id.in_({key[0] for key in expected}),
            AnalyticsDocument.status.in_(["parsed", "ready"]),
        )
    )
    current = {(chunk.document_id, chunk.chunk_id): (chunk, document) for chunk, document in rows}
    if not expected.keys() <= current.keys():
        return False
    for key, passage in expected.items():
        chunk, document = current[key]
        published_at = document.published_at.isoformat() if document.published_at else None
        if (
            passage.get("text") != chunk.text[:4000]
            or passage.get("title") != document.title
            or passage.get("published_at") != published_at
        ):
            return False
    return True


async def invalidate_cached_interpretation(
    session: AsyncSession, record_id: UUID, *, reason: str
) -> None:
    await session.execute(
        update(NewsInterpretation)
        .where(
            NewsInterpretation.id == record_id,
            NewsInterpretation.status == "completed",
        )
        .values(status="failed", error=reason, updated_at=datetime.now(UTC))
    )
    await session.commit()
