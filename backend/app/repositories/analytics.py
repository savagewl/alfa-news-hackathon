from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import undefer

from app.models.analytics import AnalyticsChunk, AnalyticsDocument, AnalyticsPage
from app.services.analytics.parser import ParsedDocument


class LeaseLostError(RuntimeError):
    pass


@dataclass(frozen=True)
class DocumentClaim:
    document_id: UUID
    token: UUID
    needs_parse: bool


async def enqueue(session: AsyncSession, **values) -> tuple[AnalyticsDocument, bool]:
    identifier = uuid4()
    statement = insert(AnalyticsDocument).values(id=identifier, **values)
    inserted = await session.scalar(
        statement.on_conflict_do_nothing(index_elements=["sha256"]).returning(AnalyticsDocument.id)
    )
    record = await session.scalar(
        select(AnalyticsDocument).where(AnalyticsDocument.sha256 == values["sha256"])
    )
    await session.commit()
    return record, inserted is None


async def claim_document(factory: async_sessionmaker) -> DocumentClaim | None:
    now = datetime.now(UTC)
    async with factory.begin() as session:
        record = await session.scalar(
            select(AnalyticsDocument)
            .where(
                AnalyticsDocument.status.in_(["queued", "extracting", "indexing"]),
                or_(AnalyticsDocument.lease_until.is_(None), AnalyticsDocument.lease_until < now),
            )
            .order_by(AnalyticsDocument.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if record is None:
            return None
        record.lease_token = uuid4()
        record.lease_until = now + timedelta(seconds=120)
        needs_parse = record.chunk_count == 0
        record.status = "extracting" if needs_parse else "indexing"
        return DocumentClaim(record.id, record.lease_token, needs_parse)


async def requeue_unembedded(factory: async_sessionmaker) -> int:
    """Вернуть в очередь документы, у которых текст готов, а векторов нет (BGE был недоступен)."""

    async with factory.begin() as session:
        result = await session.execute(
            update(AnalyticsDocument)
            .where(
                AnalyticsDocument.status == "parsed",
                AnalyticsDocument.chunk_count > 0,
                AnalyticsDocument.lease_until.is_(None),
            )
            .values(status="indexing", embedding_error=None)
        )
        return result.rowcount or 0


async def owned(session: AsyncSession, claim: DocumentClaim) -> AnalyticsDocument:
    record = await session.scalar(
        select(AnalyticsDocument).where(AnalyticsDocument.id == claim.document_id).with_for_update()
    )
    if (
        record is None
        or record.lease_token != claim.token
        or record.lease_until is None
        or record.lease_until <= datetime.now(UTC)
    ):
        raise LeaseLostError("Analytics document lease lost")
    return record


async def renew_lease(factory: async_sessionmaker, claim: DocumentClaim) -> None:
    async with factory.begin() as session:
        record = await owned(session, claim)
        record.lease_until = datetime.now(UTC) + timedelta(seconds=120)


async def load_original(factory: async_sessionmaker, claim: DocumentClaim) -> bytes:
    async with factory() as session:
        return await session.scalar(
            select(AnalyticsDocument.content).where(AnalyticsDocument.id == claim.document_id)
        )


async def save_parsed(
    factory: async_sessionmaker,
    claim: DocumentClaim,
    parsed: ParsedDocument,
) -> None:
    async with factory.begin() as session:
        record = await owned(session, claim)
        if record.sha256 != parsed.sha256:
            raise ValueError("Parsed document does not match the original")
        await session.execute(delete(AnalyticsChunk).where(AnalyticsChunk.document_id == record.id))
        await session.execute(delete(AnalyticsPage).where(AnalyticsPage.document_id == record.id))
        session.add_all(
            [
                AnalyticsPage(
                    document_id=record.id, number=p.number, payload=p.model_dump(mode="json")
                )
                for p in parsed.pages
            ]
        )
        session.add_all(
            [
                AnalyticsChunk(
                    document_id=record.id,
                    **c.model_dump(),
                    search_text=f"{record.title}\n{c.section}\n{c.text}",
                )
                for c in parsed.chunks
            ]
        )
        record.page_count = len(parsed.pages)
        record.chunk_count = len(parsed.chunks)
        record.parser_version = parsed.parser_version
        record.warnings = parsed.warnings
        if record.date_origin != "provided":
            record.published_at = parsed.detected_date
            record.date_origin = "extracted" if parsed.detected_date else "unknown"
        record.error = None
        record.embedding_error = None
        record.status = "indexing"


async def embedding_inputs(
    factory: async_sessionmaker, claim: DocumentClaim
) -> list[tuple[str, str]]:
    async with factory() as session:
        rows = await session.execute(
            select(AnalyticsChunk.chunk_id, AnalyticsChunk.search_text)
            .where(AnalyticsChunk.document_id == claim.document_id)
            .order_by(AnalyticsChunk.page_number, AnalyticsChunk.chunk_id)
        )
        return list(rows.tuples())


async def finish(
    factory: async_sessionmaker,
    claim: DocumentClaim,
    *,
    vectors: list[tuple[str, list[float]]] | None = None,
    model_version: str | None = None,
    embedding_error: str | None = None,
    error: str | None = None,
) -> None:
    async with factory.begin() as session:
        record = await owned(session, claim)
        if vectors is not None:
            for chunk_id, vector in vectors:
                await session.execute(
                    update(AnalyticsChunk)
                    .where(
                        AnalyticsChunk.document_id == record.id, AnalyticsChunk.chunk_id == chunk_id
                    )
                    .values(vector=vector, embedding_model_version=model_version)
                )
            record.embedding_model_version = model_version
        record.status = "failed" if error else ("ready" if vectors is not None else "parsed")
        record.error = error
        record.embedding_error = embedding_error
        record.lease_token = None
        record.lease_until = None


async def original(session: AsyncSession, document_id: UUID) -> AnalyticsDocument | None:
    return await session.scalar(
        select(AnalyticsDocument)
        .options(undefer(AnalyticsDocument.content))
        .where(AnalyticsDocument.id == document_id)
    )
