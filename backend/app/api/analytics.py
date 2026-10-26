import hashlib
from collections.abc import AsyncIterator
from datetime import date
from pathlib import PurePosixPath
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import CurrentUser, SettingsDep, get_current_user
from app.db.session import SessionDep
from app.models.analytics import (
    AnalyticsChunk,
    AnalyticsDocument,
    AnalyticsPage,
    BankAnalyticsSelection,
)
from app.repositories.analytics import enqueue, original
from app.schemas.analytics import (
    BankDocumentsSelection,
    DocumentResponse,
    DocumentStatus,
    DocumentUpdate,
    PageResponse,
    PassageResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)
from app.services.analytics.search import SemanticSearchUnavailable, search_analytics
from app.services.analytics.selection import effective_bank_settings
from app.services.llm.config import ServicesSettings, get_services_settings

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
    dependencies=[Depends(get_current_user)],
    responses={401: {"description": "Требуется JWT"}, 503: {"description": "Сервис недоступен"}},
)


async def analytics_session(session: SessionDep) -> AsyncIterator[AsyncSession]:
    try:
        yield session
    except (SQLAlchemyError, OSError, TimeoutError):
        await session.rollback()
        raise HTTPException(503, "Analytics database unavailable") from None


AnalyticsSession = Annotated[AsyncSession, Depends(analytics_session)]


@router.get(
    "/bank-documents",
    response_model=BankDocumentsSelection,
    summary="Документы, применяемые как аналитика банка",
)
async def get_bank_documents(
    session: AnalyticsSession,
    settings: Annotated[ServicesSettings, Depends(get_services_settings)],
) -> BankDocumentsSelection:
    effective = await effective_bank_settings(session, settings)
    return BankDocumentsSelection(document_ids=effective.bank_document_ids)


@router.put(
    "/bank-documents",
    response_model=BankDocumentsSelection,
    summary="Применить выбранные документы аналитики банка без перезапуска",
    description=(
        "Заменяет общий для всех пользователей набор аналитики банка. "
        "Нужны UUID загруженных и обработанных PDF с датой публикации. "
        "Пустой список отключает аналитику. Выбор сохраняется в БД, имеет приоритет "
        "над AI_BANK_DOCUMENT_IDS и используется в новых запросах и фоновых оценках. "
        "Обновлённые оценки появляются при следующем обращении к ленте."
    ),
    responses={
        404: {"description": "Документ не найден"},
        409: {"description": "Документ ещё не обработан или не имеет даты"},
    },
)
async def set_bank_documents(
    payload: BankDocumentsSelection, session: AnalyticsSession
) -> BankDocumentsSelection:
    ids = list(dict.fromkeys(payload.document_ids))
    if ids:
        documents = (
            await session.scalars(
                select(AnalyticsDocument)
                .where(AnalyticsDocument.id.in_(ids))
                .with_for_update()
            )
        ).all()
        if set(ids) != {document.id for document in documents}:
            raise HTTPException(404, "Analytics document not found")
        if any(
            document.status not in {"parsed", "ready"}
            or not document.chunk_count
            or document.published_at is None
            for document in documents
        ):
            raise HTTPException(409, "Documents must be processed and have a publication date")
    values = [str(value) for value in ids]
    statement = insert(BankAnalyticsSelection).values(id=1, document_ids=values)
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=[BankAnalyticsSelection.id], set_={"document_ids": values}
        )
    )
    await session.commit()
    return BankDocumentsSelection(document_ids=ids)


async def require_document(session: AsyncSession, document_id: UUID) -> AnalyticsDocument:
    record = await session.get(AnalyticsDocument, document_id)
    if record is None:
        raise HTTPException(404, "Analytics document not found")
    return record


@router.post(
    "/documents",
    response_model=UploadResponse,
    status_code=202,
    summary="Загрузить аналитический PDF",
    responses={
        413: {"description": "Превышен размер PDF"},
        422: {"description": "Неверный формат"},
    },
)
async def upload_document(
    session: AnalyticsSession,
    user: CurrentUser,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="PDF, включая сканы на русском и английском")],
    publisher: Annotated[str, Form(min_length=1, max_length=300)],
    title: Annotated[str | None, Form(min_length=1, max_length=500)] = None,
    published_at: Annotated[date | None, Form()] = None,
    topics: Annotated[str, Form(max_length=2000, description="Темы через запятую")] = "",
) -> UploadResponse:
    """Общий корпус для авторизованных пользователей. Издатель задаётся загрузившим.

    Возвращает 202 и status_url. Проверка PDF/OCR/BGE выполняется в фоне; ошибки — в статусе.
    Один и тот же файл не дублируется. Его повторная загрузка не меняет метаданные.
    Загруженный материал не становится автоматически подтверждённой позицией банка.
    """
    try:
        content = await file.read(settings.analytics_max_bytes + 1)
    finally:
        await file.close()
    if len(content) > settings.analytics_max_bytes:
        raise HTTPException(413, "PDF file is too large")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(422, "Expected a PDF file")
    publisher = publisher.strip()
    if not publisher or (title is not None and not title.strip()):
        raise HTTPException(422, "Publisher and title must not be blank")
    filename = PurePosixPath((file.filename or "upload.pdf").replace("\\", "/")).name[:500]
    record, duplicate = await enqueue(
        session,
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        filename=filename,
        title=title.strip() if title else filename.removesuffix(".pdf"),
        publisher=publisher,
        published_at=published_at,
        date_origin="provided" if published_at else "unknown",
        uploaded_by=user.user_id,
        topics=list(dict.fromkeys(t.strip()[:100] for t in topics.split(",") if t.strip()))[:30],
    )
    return UploadResponse(
        document=DocumentResponse.model_validate(record),
        duplicate=duplicate,
        status_url=f"/analytics/documents/{record.id}",
    )


@router.get("/documents", response_model=list[DocumentResponse], summary="Каталог аналитики")
async def list_documents(
    session: AnalyticsSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    publisher: Annotated[str | None, Query(max_length=300)] = None,
    status: DocumentStatus | None = None,
) -> list[DocumentResponse]:
    query = select(AnalyticsDocument)
    if publisher:
        query = query.where(AnalyticsDocument.publisher == publisher)
    if status:
        query = query.where(AnalyticsDocument.status == status)
    records = (
        await session.scalars(
            query.order_by(AnalyticsDocument.created_at.desc(), AnalyticsDocument.id)
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return [DocumentResponse.model_validate(r) for r in records]


@router.get(
    "/documents/{document_id}", response_model=DocumentResponse, summary="Статус обработки PDF"
)
async def get_document(document_id: UUID, session: AnalyticsSession) -> DocumentResponse:
    return DocumentResponse.model_validate(await require_document(session, document_id))


@router.patch(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    summary="Уточнить дату, название, издателя и темы",
)
async def update_document(
    document_id: UUID,
    payload: DocumentUpdate,
    session: AnalyticsSession,
) -> DocumentResponse:
    record = await session.scalar(
        select(AnalyticsDocument).where(AnalyticsDocument.id == document_id).with_for_update()
    )
    if record is None:
        raise HTTPException(404, "Analytics document not found")
    if record.status in {"queued", "extracting", "indexing"}:
        raise HTTPException(409, "Wait for document processing to finish")
    for name, value in payload.model_dump(exclude_unset=True).items():
        setattr(record, name, value.strip() if isinstance(value, str) else value)
    if "published_at" in payload.model_fields_set:
        record.date_origin = "provided" if record.published_at else "unknown"
        if record.published_at:
            record.warnings = [
                w for w in record.warnings if not w.startswith("Дата публикации не определена")
            ]
    if "title" in payload.model_fields_set and record.chunk_count:
        # Search text includes the title, so its embeddings must be rebuilt as well.
        await session.execute(
            update(AnalyticsChunk)
            .where(AnalyticsChunk.document_id == document_id)
            .values(
                search_text=record.title
                + "\n"
                + AnalyticsChunk.section
                + "\n"
                + AnalyticsChunk.text,
                vector=None,
                embedding_model_version=None,
            )
        )
        record.embedding_model_version = None
        record.status = "indexing"
    await session.commit()
    return DocumentResponse.model_validate(record)


@router.get(
    "/documents/{document_id}/passages",
    response_model=list[PassageResponse],
    summary="Извлечённые фрагменты со ссылками на страницы",
)
async def list_passages(
    document_id: UUID,
    session: AnalyticsSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[PassageResponse]:
    await require_document(session, document_id)
    records = (
        await session.scalars(
            select(AnalyticsChunk)
            .where(AnalyticsChunk.document_id == document_id)
            .order_by(AnalyticsChunk.page_number, AnalyticsChunk.chunk_id)
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return [PassageResponse.model_validate(r) for r in records]


@router.get(
    "/documents/{document_id}/pages/{number}",
    response_model=PageResponse,
    summary="Распознанная страница и координаты блоков",
)
async def get_page(document_id: UUID, number: int, session: AnalyticsSession) -> PageResponse:
    record = await session.get(AnalyticsPage, (document_id, number))
    if record is None:
        raise HTTPException(404, "Analytics page not found")
    return PageResponse(document_id=document_id, page=record.payload)


@router.get("/documents/{document_id}/file", summary="Открыть исходный PDF")
async def get_file(document_id: UUID, session: AnalyticsSession) -> Response:
    record = await original(session, document_id)
    if record is None:
        raise HTTPException(404, "Analytics document not found")
    return Response(
        record.content,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{document_id}.pdf"'},
    )


@router.post(
    "/documents/{document_id}/retry",
    response_model=DocumentResponse,
    status_code=202,
    summary="Повторить обработку или векторизацию",
)
async def retry_document(document_id: UUID, session: AnalyticsSession) -> DocumentResponse:
    record = await session.scalar(
        select(AnalyticsDocument).where(AnalyticsDocument.id == document_id).with_for_update()
    )
    if record is None:
        raise HTTPException(404, "Analytics document not found")
    if record.status not in {"failed", "parsed", "ready"}:
        raise HTTPException(409, "Document is already queued or processing")
    record.status = "indexing" if record.chunk_count else "queued"
    record.lease_token = None
    record.lease_until = None
    record.error = None
    record.embedding_error = None
    await session.commit()
    return DocumentResponse.model_validate(record)


@router.post(
    "/search", response_model=SearchResponse, summary="Найти источники для AI-интерпретации"
)
async def search(request: SearchRequest, session: AnalyticsSession) -> SearchResponse:
    """Текстовый, векторный или гибридный поиск. as_of исключает более поздние и недатированные PDF.

    Возвращает текст, атрибуцию, страницу, evidence_ref и ссылку на оригинал.
    score — относительный RRF-балл ранжирования, не вероятность достоверности.
    """
    try:
        return await search_analytics(session, request)
    except SemanticSearchUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
