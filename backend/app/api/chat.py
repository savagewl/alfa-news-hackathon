import asyncio
import logging
import time
from typing import Annotated
from uuid import UUID

import httpx2 as httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import TypeAdapter
from sqlalchemy.exc import SQLAlchemyError

from app.api.company import CompanyId
from app.api.company_news import AISettings, NewsSession, _run_in_llm_slot
from app.api.dependencies import CurrentUser, get_current_user
from app.models.news import NewsArticle
from app.repositories.chat import (
    ChatThreadNotFound,
    complete_answer,
    create_thread,
    fail_answer,
    get_owned_thread,
    list_messages,
    reserve_exchange,
)
from app.repositories.company import CompanyRepository
from app.repositories.interpretations import find_current_interpretation
from app.repositories.news_embeddings import fetch_news_event, news_input_fingerprint
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatMessageResult,
    ChatSource,
    ChatThreadResponse,
)
from app.schemas.company_news import InterpretationSource
from app.schemas.news import NewsError
from app.services.chat import InvalidChatAnswer, bounded_history, generate_chat_answer
from app.services.company_news import (
    LLMUnavailable,
    company_input_fingerprint,
    load_company_context,
    model_prompt_version,
)
from app.services.llm.schemas import BankPassage, InsightDraft

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/companies",
    tags=["Company news chat"],
    dependencies=[Depends(get_current_user)],
    responses={
        401: {"model": NewsError, "description": "Требуется JWT"},
        404: {
            "model": NewsError,
            "description": "Компания, новость или принадлежащий пользователю чат не найдены",
        },
        503: {"model": NewsError, "description": "БД или LLM недоступна"},
    },
)


def _message_response(message) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=message.id,
        thread_id=message.thread_id,
        client_message_id=message.client_message_id,
        reply_to_id=message.reply_to_id,
        role=message.role,
        status=message.status,
        content=message.content,
        answer_kind=message.answer_kind,
        sources=TypeAdapter(list[ChatSource]).validate_python(message.sources or []),
        warnings=list(message.warnings or []),
        created_at=message.created_at,
    )


async def _context(
    session: NewsSession,
    *,
    user: CurrentUser,
    company_id: str,
    news_id: UUID,
):
    company = await CompanyRepository(session, user.user_id).get(company_id)
    article = await session.get(NewsArticle, news_id)
    if article is None:
        raise HTTPException(404, "News article not found")
    return company, article


@router.post(
    "/{company_id}/news/{news_id}/chat/threads",
    response_model=ChatThreadResponse,
    status_code=201,
    summary="Создать личный чат по новости компании",
)
async def start_news_chat(
    company_id: CompanyId,
    news_id: UUID,
    session: NewsSession,
    user: CurrentUser,
) -> ChatThreadResponse:
    """Чат принадлежит текущему пользователю и недоступен другим участникам компании."""

    await _context(session, user=user, company_id=company_id, news_id=news_id)
    thread = await create_thread(
        session, user_id=user.user_id, company_id=company_id, news_id=news_id
    )
    return ChatThreadResponse.model_validate(thread)


@router.get(
    "/{company_id}/news/{news_id}/chat/threads/{thread_id}/messages",
    response_model=ChatHistoryResponse,
    summary="Получить сохранённую историю личного чата",
)
async def news_chat_history(
    company_id: CompanyId,
    news_id: UUID,
    thread_id: UUID,
    session: NewsSession,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    before: UUID | None = None,
) -> ChatHistoryResponse:
    """Только чтение БД: запрос не запускает LLM, поиск или новую интерпретацию."""

    await CompanyRepository(session, user.user_id).get(company_id)
    try:
        thread = await get_owned_thread(
            session,
            thread_id=thread_id,
            user_id=user.user_id,
            company_id=company_id,
            news_id=news_id,
        )
    except ChatThreadNotFound:
        raise HTTPException(404, "Chat thread not found") from None
    try:
        messages, has_more = await list_messages(
            session, thread.id, limit=limit, before=before
        )
    except ChatThreadNotFound:
        raise HTTPException(404, "Chat history cursor not found") from None
    return ChatHistoryResponse(
        thread=ChatThreadResponse.model_validate(thread),
        messages=[_message_response(message) for message in messages],
        limit=limit,
        has_more=has_more,
        next_before=messages[0].id if has_more and messages else None,
    )


@router.post(
    "/{company_id}/news/{news_id}/chat/threads/{thread_id}/messages",
    response_model=ChatMessageResult,
    summary="Задать вопрос по готовой интерпретации",
    responses={
        202: {
            "model": ChatMessageResult,
            "description": "Ответ на тот же client_message_id уже формируется",
        },
        409: {
            "model": NewsError,
            "description": "Актуальная подтверждённая интерпретация ещё не готова",
        },
        429: {"model": NewsError, "description": "Все слоты LLM заняты"},
        502: {"model": NewsError, "description": "LLM вернула непроверяемый ответ"},
        504: {"model": NewsError, "description": "Истёк общий срок ответа"},
    },
)
async def send_news_chat_message(
    company_id: CompanyId,
    news_id: UUID,
    thread_id: UUID,
    body: ChatMessageRequest,
    session: NewsSession,
    user: CurrentUser,
    ai: AISettings,
    request: Request,
    response: Response,
) -> ChatMessageResult:
    """Использует только уже подтверждённый инсайт; не создаёт его автоматически."""

    deadline = time.monotonic() + ai.interpretation_timeout_seconds
    company, article = await _context(
        session, user=user, company_id=company_id, news_id=news_id
    )
    try:
        thread = await get_owned_thread(
            session,
            thread_id=thread_id,
            user_id=user.user_id,
            company_id=company_id,
            news_id=news_id,
        )
    except ChatThreadNotFound:
        raise HTTPException(404, "Chat thread not found") from None

    context = await load_company_context(session, company)
    cached = await find_current_interpretation(
        session,
        company_id=company_id,
        news_id=news_id,
        company_input_fingerprint=company_input_fingerprint(context),
        news_input_fingerprint=news_input_fingerprint(article, ai),
        model_prompt_version=model_prompt_version(ai),
        trusted_bank_document_ids=ai.bank_document_ids,
    )
    if cached is None:
        raise HTTPException(
            409,
            "Approved current interpretation is not ready; request the interpretation first",
        )
    payload = cached.result or {}
    try:
        insight = InsightDraft.model_validate(payload["interpretation"])
        cached_sources = TypeAdapter(list[InterpretationSource]).validate_python(
            payload.get("sources") or []
        )
        cached_passages = TypeAdapter(list[BankPassage]).validate_python(
            payload.get("bank_passages") or []
        )
    except (KeyError, ValueError):
        raise HTTPException(409, "Approved current interpretation is unavailable") from None
    news, prep_warnings, _ = await fetch_news_event(session, article, settings=ai)
    sources = cached_sources
    passages = cached_passages

    try:
        user_message, assistant, token, reused = await reserve_exchange(
            session,
            thread_id=thread.id,
            user_id=user.user_id,
            client_message_id=body.client_message_id,
            question=body.question.strip(),
            lease_seconds=ai.interpretation_timeout_seconds,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    if token is None:
        if assistant.status == "generating":
            response.status_code = 202
            response.headers["Retry-After"] = "2"
        return ChatMessageResult(
            user_message=_message_response(user_message),
            assistant_message=_message_response(assistant),
            reused=True,
        )

    messages, _ = await list_messages(session, thread.id, limit=100)
    history = bounded_history(messages, exclude_message_id=user_message.id)
    await session.commit()
    slots = getattr(request.app.state, "interpretation_slots", None)
    if slots is None:
        slots = request.app.state.interpretation_slots = asyncio.Semaphore(
            ai.interpretation_concurrency
        )
    try:
        answer, answer_kind, answer_sources, answer_warnings = await _run_in_llm_slot(
            slots,
            deadline,
            generate_chat_answer,
            context,
            news,
            insight,
            passages,
            sources,
            body.question,
            history,
            ai,
        )
        saved = await complete_answer(
            session,
            message_id=assistant.id,
            token=token,
            content=answer,
            answer_kind=answer_kind,
            sources=answer_sources,
            warnings=list(dict.fromkeys(prep_warnings + answer_warnings)),
        )
        if saved is None:
            raise HTTPException(409, "Chat answer lease changed; retry with the same id")
        return ChatMessageResult(
            user_message=_message_response(user_message),
            assistant_message=_message_response(saved),
            reused=reused,
        )
    except LLMUnavailable as exc:
        await fail_answer(
            session, message_id=assistant.id, token=token, warning="LLM is not configured"
        )
        raise HTTPException(503, str(exc)) from None
    except TimeoutError:
        await fail_answer(
            session, message_id=assistant.id, token=token, warning="LLM deadline exceeded"
        )
        raise HTTPException(504, "Chat answer timed out; retry later") from None
    except InvalidChatAnswer:
        await fail_answer(
            session,
            message_id=assistant.id,
            token=token,
            warning="LLM answer has no valid evidence",
        )
        raise HTTPException(502, "LLM returned an answer without valid evidence") from None
    except HTTPException:
        await fail_answer(
            session, message_id=assistant.id, token=token, warning="LLM request was not started"
        )
        raise
    except (RuntimeError, ValueError, httpx.HTTPError, SQLAlchemyError):
        logger.exception("Chat answer failed thread=%s", thread_id)
        await fail_answer(
            session, message_id=assistant.id, token=token, warning="Chat answer failed"
        )
        raise HTTPException(502, "Chat answer failed; retry later") from None
