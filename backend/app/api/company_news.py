import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

import httpx2 as httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import AwareDatetime
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from app.api.company import CompanyId, Limit, Offset
from app.api.dependencies import CurrentUser, get_current_user
from app.db.session import SessionDep
from app.models.company import Company
from app.models.news import NewsArticle
from app.repositories.assessments import (
    claim_assessment,
    complete_assessment,
    fail_assessment,
    load_assessment_states,
)
from app.repositories.company import CompanyNotFoundError, CompanyRepository
from app.repositories.company_news import list_company_news
from app.repositories.interpretations import (
    acquire_interpretation,
    complete_interpretation,
    find_current_interpretation,
    interpretation_evidence_is_current,
    invalidate_cached_interpretation,
)
from app.repositories.news_embeddings import news_input_fingerprint
from app.schemas.company_news import (
    AssessmentStreamRequest,
    CompanyNewsPage,
    InterpretationResponse,
)
from app.schemas.news import NewsError, NewsTopic
from app.services.analytics.selection import effective_bank_settings
from app.services.assessments import (
    assessment_fingerprint,
    assessment_view,
    build_assessment,
    llm_is_configured,
)
from app.services.company_news import (
    LLMUnavailable,
    company_input_fingerprint,
    completed_response,
    emit_interpretation_progress,
    final_cache_key,
    generate_interpretation,
    interpretation_progress,
    load_company_context,
    model_prompt_version,
    prepare_interpretation,
    require_llm,
    retrieval_fingerprint,
    retrieve_for_verifier,
    validate_result,
)
from app.services.llm.config import ServicesSettings, get_services_settings
from app.services.llm.provider import LLMBudgetExceededError

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/companies",
    tags=["Company news"],
    dependencies=[Depends(get_current_user)],
    responses={
        401: {"model": NewsError, "description": "Требуется JWT"},
        404: {"model": NewsError, "description": "Компания недоступна или новость не найдена"},
        503: {"model": NewsError, "description": "БД или LLM недоступна"},
    },
)


async def company_news_session(session: SessionDep) -> AsyncIterator[AsyncSession]:
    try:
        yield session
    except CompanyNotFoundError:
        await session.rollback()
        raise HTTPException(404, "Company not found") from None
    except (SQLAlchemyError, OSError):
        await session.rollback()
        logger.exception("Company news database operation failed")
        raise HTTPException(503, "Database unavailable") from None


NewsSession = Annotated[AsyncSession, Depends(company_news_session)]


async def get_effective_ai_settings(
    session: NewsSession,
    settings: Annotated[ServicesSettings, Depends(get_services_settings)],
) -> ServicesSettings:
    return await effective_bank_settings(session, settings)


AISettings = Annotated[ServicesSettings, Depends(get_effective_ai_settings)]


async def _run_in_llm_slot(slots: asyncio.Semaphore, deadline: float, function, *args):
    """Keep capacity reserved until a timed-out worker thread really exits."""

    try:
        async with asyncio.timeout(0.05):
            await slots.acquire()
    except TimeoutError:
        raise HTTPException(
            429, "LLM is busy; retry later", headers={"Retry-After": "3"}
        ) from None
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        result = await asyncio.wait_for(asyncio.shield(task), remaining)
    except (TimeoutError, asyncio.CancelledError):
        def release_when_done(done: asyncio.Task) -> None:
            try:
                done.exception()
            except (asyncio.CancelledError, Exception):
                pass
            slots.release()

        task.add_done_callback(release_when_done)
        raise
    except Exception:
        slots.release()
        raise
    else:
        slots.release()
        return result


@router.get(
    "/{company_id}/news",
    response_model=CompanyNewsPage,
    summary="Быстрая новостная сводка компании",
)
async def company_news(
    company_id: CompanyId,
    session: NewsSession,
    user: CurrentUser,
    ai: AISettings,
    limit: Limit = 20,
    offset: Offset = 0,
    source: Annotated[str | None, Query(max_length=100)] = None,
    topic: NewsTopic | None = None,
    published_from: AwareDatetime | None = None,
    published_to: AwareDatetime | None = None,
    relevant_only: bool = True,
) -> CompanyNewsPage:
    """Быстрая сводка из сохранённых данных; запрос не вызывает LLM или сервис эмбеддингов.

    Готовые актуальные AI-анонсы и профиль улучшают ранжирование. Если они ещё не готовы,
    используются RSS-анонс, заголовок и явные правила. Семантические кандидаты выбираются
    только после source/topic/date фильтров; совпадения по правилам не ограничиваются этим
    окном. Риски и возможности возвращаются лишь из готовой подтверждённой интерпретации.
    relevant_only=false включает новости без совпадений. Даты включительные, с timezone.
    """
    company = await CompanyRepository(session, user.user_id).get(company_id)
    if published_from and published_to and published_from > published_to:
        raise HTTPException(422, "Period start must precede period end")
    return await list_company_news(
        session,
        company,
        limit=limit,
        offset=offset,
        source=source,
        topic=topic,
        published_from=published_from,
        published_to=published_to,
        relevant_only=relevant_only,
        settings=ai,
    )


@router.post(
    "/{company_id}/news/{news_id}/interpretation",
    response_model=InterpretationResponse,
    summary="Сгенерировать или получить сохранённую AI-интерпретацию новости",
    responses={
        202: {
            "model": InterpretationResponse,
            "description": "Другой запрос уже генерирует результат; повторите этот POST",
        },
        429: {"model": NewsError, "description": "Все слоты LLM заняты; повторите позже"},
        409: {
            "model": NewsError,
            "description": "Генерацию перехватил другой запрос; повторите POST",
        },
        502: {"model": NewsError, "description": "Ошибка ответа LLM"},
        504: {"model": NewsError, "description": "Истёк общий срок генерации"},
    },
)
async def interpret_news(
    company_id: CompanyId,
    news_id: UUID,
    session: NewsSession,
    user: CurrentUser,
    ai: AISettings,
    request: Request,
    response: Response,
) -> InterpretationResponse:
    """Первый запрос ожидает LLM, повторный использует кеш. Тела запроса нет.

    Доступ — только участникам компании. Аналитика банка выбирается через /analytics/bank-documents;
    без подтверждённых материалов bank_position не формируется. Непроверенный черновик
    не публикуется: status=needs_revision/insufficient, interpretation=null.
    При изменении входных данных или истечении TTL результат генерируется заново.
    """
    deadline = time.monotonic() + ai.interpretation_timeout_seconds
    company = await CompanyRepository(session, user.user_id).get(company_id)
    article = await session.get(NewsArticle, news_id)
    if article is None:
        raise HTTPException(404, "News article not found")
    cached = await find_current_interpretation(
        session,
        company_id=company_id,
        news_id=news_id,
        company_input_fingerprint=company_input_fingerprint(
            await load_company_context(session, company)
        ),
        news_input_fingerprint=news_input_fingerprint(article, ai),
        model_prompt_version=model_prompt_version(ai),
        trusted_bank_document_ids=ai.bank_document_ids,
    )
    if cached is not None:
        return InterpretationResponse.model_validate({**cached.result, "reused": True})
    try:
        async with asyncio.timeout(max(deadline - time.monotonic(), 0.001)):
            prepared = await prepare_interpretation(session, company, article, ai)
    except TimeoutError:
        raise HTTPException(504, "Evidence retrieval timed out; retry later") from None
    context, news, passages, sources, warnings, cache_key = prepared
    company_fingerprint = company_input_fingerprint(context)
    news_fingerprint = news_input_fingerprint(article, ai)
    prompt_version = model_prompt_version(ai)
    # End all read transactions before network/LLM work, releasing the pooled connection.
    await session.commit()
    record, token = await acquire_interpretation(
        session,
        company_id=company_id,
        news_id=news_id,
        request_fingerprint=cache_key,
        company_input_fingerprint=company_fingerprint,
        news_input_fingerprint=news_fingerprint,
        model_prompt_version=prompt_version,
        timeout_seconds=ai.interpretation_timeout_seconds,
        cache_hours=ai.interpretation_cache_hours,
    )
    if token is None:
        if record.result:
            if await interpretation_evidence_is_current(
                session,
                record.result,
                trusted_bank_document_ids=ai.bank_document_ids,
            ):
                return InterpretationResponse.model_validate({**record.result, "reused": True})
            await invalidate_cached_interpretation(
                session, record.id, reason="Stored bank evidence changed"
            )
            record, token = await acquire_interpretation(
                session,
                company_id=company_id,
                news_id=news_id,
                request_fingerprint=cache_key,
                company_input_fingerprint=company_fingerprint,
                news_input_fingerprint=news_fingerprint,
                model_prompt_version=prompt_version,
                timeout_seconds=ai.interpretation_timeout_seconds,
                cache_hours=ai.interpretation_cache_hours,
            )
        if token is None:
            response.status_code = 202
            response.headers["Retry-After"] = "2"
            return InterpretationResponse(
                id=record.id,
                company_id=company_id,
                news_id=news_id,
                status="generating",
                model=ai.llm_model,
                is_demo=ai.llm_mock,
                reused=True,
            )
    try:
        require_llm(ai)
    except LLMUnavailable as exc:
        await complete_interpretation(
            session, record.id, token, result=None, error="LLM is not configured"
        )
        raise HTTPException(503, str(exc)) from None
    slots = getattr(request.app.state, "interpretation_slots", None)
    if slots is None:
        slots = request.app.state.interpretation_slots = asyncio.Semaphore(
            ai.interpretation_concurrency
        )
    try:
        try:
            result = await _run_in_llm_slot(
                slots,
                deadline,
                generate_interpretation,
                context,
                news,
                passages,
                ai,
            )
        except HTTPException:
            await complete_interpretation(
                session, record.id, token, result=None, error="LLM concurrency limit"
            )
            raise
        if result.verification.status != "APPROVED" and result.verification.retry_queries:
            emit_interpretation_progress("progress", {"stage": "retrieving"})
            before = retrieval_fingerprint(passages)
            passages, sources, retry_warnings = await retrieve_for_verifier(
                session,
                queries=result.verification.retry_queries,
                settings=ai,
                article=article,
                passages=passages,
                sources=sources,
            )
            warnings.extend(retry_warnings)
            if retrieval_fingerprint(passages) != before:
                await session.commit()
                result = await _run_in_llm_slot(
                    slots,
                    deadline,
                    generate_interpretation,
                    context,
                    news,
                    passages,
                    ai,
                )
        result = validate_result(result, context, news, sources)
        payload = completed_response(record.id, company_id, news_id, result, sources, warnings, ai)
        stored_payload = payload.model_dump(mode="json")
        stored_payload["bank_passages"] = [
            passage.model_dump(mode="json") for passage in passages
        ]
        saved = await complete_interpretation(
            session,
            record.id,
            token,
            result=stored_payload,
            final_cache_key=final_cache_key(cache_key, passages),
            retrieval_fingerprint=retrieval_fingerprint(passages),
        )
        if not saved:
            raise HTTPException(409, "Interpretation lease changed; retry the request")
        return payload
    except TimeoutError:
        await complete_interpretation(
            session, record.id, token, result=None, error="LLM deadline exceeded"
        )
        raise HTTPException(504, "Interpretation timed out; retry later") from None
    except asyncio.CancelledError:
        await complete_interpretation(
            session, record.id, token, result=None, error="Request cancelled"
        )
        raise
    except (RuntimeError, ValueError, httpx.HTTPError):
        logger.exception("Interpretation generation failed company=%s news=%s", company_id, news_id)
        await complete_interpretation(
            session, record.id, token, result=None, error="LLM generation failed"
        )
        raise HTTPException(502, "LLM generation failed; retry later") from None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@router.post(
    "/{company_id}/news/{news_id}/interpretation/stream",
    summary="Постепенно сформировать подробную AI-интерпретацию",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE progress, draft snapshots, final result, done",
        }
    },
)
async def stream_news_interpretation(
    company_id: CompanyId,
    news_id: UUID,
    session: NewsSession,
    user: CurrentUser,
    ai: AISettings,
    request: Request,
) -> StreamingResponse:
    """Stream real LLM stages; draft snapshots are explicitly unverified."""

    await CompanyRepository(session, user.user_id).get(company_id)
    if await session.get(NewsArticle, news_id) is None:
        raise HTTPException(404, "News article not found")
    await session.commit()
    factory = request.app.state.database.session_factory

    async def event_stream():
        yield _sse("progress", {"stage": "preparing"})
        loop = asyncio.get_running_loop()
        # Model output is bounded; an unbounded per-request queue preserves every JSON field
        # delta without blocking the provider thread or dropping UI state transitions.
        events: asyncio.Queue = asyncio.Queue()
        closed = threading.Event()
        next_revision_version = 0
        active_revision_version: int | None = None

        def progress_callback(event: str, data: dict) -> None:
            nonlocal active_revision_version, next_revision_version
            if closed.is_set():
                return
            payload = dict(data)
            if event == "reset":
                active_revision_version = next_revision_version
                next_revision_version += 1
            elif event in {"delta", "draft"} and active_revision_version is None:
                active_revision_version = next_revision_version
                next_revision_version += 1
            if event in {"reset", "delta", "draft"}:
                payload["revision_version"] = active_revision_version
                payload["verified"] = False

            def enqueue() -> None:
                if closed.is_set():
                    return
                events.put_nowait((event, payload))

            try:
                loop.call_soon_threadsafe(enqueue)
            except RuntimeError:
                return

        async def run() -> None:
            deadline = time.monotonic() + ai.interpretation_timeout_seconds + 30
            waiting_emitted = False
            try:
                with interpretation_progress(progress_callback):
                    while True:
                        async with factory() as item_session:
                            item_response = Response()
                            payload = await interpret_news(
                                company_id,
                                news_id,
                                item_session,
                                user,
                                ai,
                                request,
                                item_response,
                            )
                        if item_response.status_code != 202:
                            await asyncio.sleep(0)
                            await events.put(
                                ("result", payload.model_dump(mode="json"))
                            )
                            return
                        if not waiting_emitted:
                            progress_callback("progress", {"stage": "waiting"})
                            waiting_emitted = True
                        if time.monotonic() >= deadline:
                            await events.put(
                                (
                                    "error",
                                    {
                                        "detail": (
                                            "Interpretation is still generating; "
                                            "retry stream"
                                        ),
                                        "http_status": 504,
                                        "retryable": True,
                                    },
                                )
                            )
                            return
                        await asyncio.sleep(2)
            except HTTPException as exc:
                await events.put(
                    (
                        "error",
                        {
                            "detail": str(exc.detail),
                            "http_status": exc.status_code,
                            "retryable": exc.status_code in {409, 429, 502, 503, 504},
                        },
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Interpretation stream failed news=%s", news_id)
                await events.put(
                    (
                        "error",
                        {
                            "detail": "Interpretation generation failed",
                            "retryable": True,
                        },
                    )
                )

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(events.get(), timeout=15)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield _sse(event, data)
                if event in {"result", "error"}:
                    break
            yield _sse("done", {})
        finally:
            closed.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _short_patch(news_id: UUID, payload: dict, *, reused: bool) -> dict:
    """Card patch from a short assessment; empty risks/opportunities are null (not rendered)."""

    view = assessment_view(payload)
    return {
        "news_id": str(news_id),
        "status": "ready",
        "warnings": payload.get("warnings") or [],
        "is_demo": bool(payload.get("is_demo")),
        "reused": reused,
        "brief_impact": view["brief_impact"],
        "risks": view["risks"],
        "opportunities": view["opportunities"],
        "bank_position": view["bank_analytics"],
        "sources": view["bank_sources"],
    }


def _failed_patch(news_id: UUID, detail: str, *, http_status: int | None = None) -> dict:
    patch = {"news_id": str(news_id), "status": "failed", "detail": detail, "retryable": True}
    if http_status is not None:
        patch["http_status"] = http_status
    return patch


@router.post(
    "/{company_id}/news/assessments/stream",
    summary="Постепенно сформировать короткие AI-оценки видимых новостей",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE events assessment (queued/generating/terminal) and done",
        },
        422: {"model": NewsError, "description": "Некорректный список news_ids"},
    },
)
async def stream_news_assessments(
    company_id: CompanyId,
    body: AssessmentStreamRequest,
    session: NewsSession,
    user: CurrentUser,
    ai: AISettings,
    request: Request,
) -> StreamingResponse:
    """Stream complete validated card patches; this is not token streaming.

    Каждая карточка получает короткую оценку (один вызов LLM, 2-6 секунд) вместо полной
    интерпретации с верификатором. Пустые риски и возможности приходят как null.
    Полная интерпретация формируется отдельно, при открытии события.
    """

    await CompanyRepository(session, user.user_id).get(company_id)
    found = set(
        await session.scalars(select(NewsArticle.id).where(NewsArticle.id.in_(body.news_ids)))
    )
    missing = [str(news_id) for news_id in body.news_ids if news_id not in found]
    if missing:
        raise HTTPException(404, f"News articles not found: {', '.join(missing)}")
    await session.commit()

    factory = request.app.state.database.session_factory
    slots = getattr(request.app.state, "assessment_slots", None)
    if slots is None:
        slots = request.app.state.assessment_slots = asyncio.Semaphore(ai.assessment_concurrency)

    async def ready_payload(news_id: UUID, fingerprint: str) -> dict | None:
        async with factory() as item_session:
            ready, _ = await load_assessment_states(
                item_session,
                company_id,
                {news_id: fingerprint},
                trusted_bank_document_ids=ai.bank_document_ids,
            )
        return ready.get(news_id)

    async def generate(news_id: UUID, fingerprint: str) -> dict | None:
        """Payload if this request generated it; None when another worker holds the lease."""

        async with factory() as claim_session:
            claim = await claim_assessment(
                claim_session,
                company_id,
                news_id,
                fingerprint,
                lease_seconds=ai.assessment_lease_seconds,
            )
        if claim is None:
            return None
        try:
            async with factory() as item_session:
                company = await item_session.get(Company, company_id)
                article = await item_session.get(NewsArticle, news_id)
                deadline = time.monotonic() + ai.assessment_lease_seconds - 15
                fingerprint, payload = await build_assessment(
                    item_session, company, article, ai, deadline=deadline
                )
            async with factory() as save_session:
                saved = await complete_assessment(
                    save_session,
                    claim,
                    fingerprint=fingerprint,
                    result=payload,
                    cache_hours=ai.assessment_cache_hours,
                )
            if not saved:
                raise RuntimeError("Assessment lease changed; retry the request")
            return payload
        except BaseException as exc:
            async with factory() as fail_session:
                await fail_assessment(
                    fail_session,
                    claim,
                    error=f"{type(exc).__name__}: {exc}",
                    retry_seconds=ai.assessment_retry_seconds,
                    max_attempts=ai.assessment_max_attempts,
                )
            raise

    async def run_one(news_id: UUID, events: asyncio.Queue) -> None:
        try:
            async with factory() as item_session:
                article = await item_session.get(NewsArticle, news_id)
                company = await item_session.get(Company, company_id)
                fingerprint = assessment_fingerprint(
                    company_input_fingerprint(await load_company_context(item_session, company)),
                    article,
                    ai,
                )
            payload = await ready_payload(news_id, fingerprint)
            if payload is not None:
                await events.put(("assessment", _short_patch(news_id, payload, reused=True)))
                return
            if not llm_is_configured(ai):
                await events.put(
                    ("assessment", _failed_patch(news_id, "LLM is not configured", http_status=503))
                )
                return
            deadline = time.monotonic() + ai.assessment_lease_seconds
            while True:
                async with slots:
                    payload = await ready_payload(news_id, fingerprint)
                    if payload is not None:
                        await events.put(
                            ("assessment", _short_patch(news_id, payload, reused=True))
                        )
                        return
                    await events.put(
                        ("assessment", {"news_id": str(news_id), "status": "generating"})
                    )
                    payload = await generate(news_id, fingerprint)
                if payload is not None:
                    await events.put(("assessment", _short_patch(news_id, payload, reused=False)))
                    return
                # Генерацию ведёт фоновый воркер или другой запрос — ждём готовый результат.
                await asyncio.sleep(1.5)
                payload = await ready_payload(news_id, fingerprint)
                if payload is not None:
                    await events.put(("assessment", _short_patch(news_id, payload, reused=True)))
                    return
                if time.monotonic() >= deadline:
                    await events.put(
                        ("assessment", _failed_patch(news_id, "Assessment is still generating"))
                    )
                    return
        except LLMBudgetExceededError as exc:
            await events.put(("assessment", _failed_patch(news_id, str(exc), http_status=429)))
        except Exception:
            logger.exception("Assessment stream item failed news=%s", news_id)
            await events.put(("assessment", _failed_patch(news_id, "Assessment generation failed")))

    async def event_stream():
        events: asyncio.Queue = asyncio.Queue()
        for news_id in body.news_ids:
            yield _sse("assessment", {"news_id": str(news_id), "status": "queued"})
        tasks = [asyncio.create_task(run_one(news_id, events)) for news_id in body.news_ids]
        counts: dict[str, int] = {}
        terminal_count = 0
        try:
            while terminal_count < len(body.news_ids):
                try:
                    event, data = await asyncio.wait_for(events.get(), timeout=15)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield _sse(event, data)
                if data["status"] == "generating":
                    continue
                terminal_count += 1
                counts[data["status"]] = counts.get(data["status"], 0) + 1
            yield _sse("done", {"total": len(body.news_ids), "counts": counts})
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
