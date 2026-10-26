from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import and_, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import NewsChatMessage, NewsChatThread


class ChatThreadNotFound(LookupError):
    pass


async def create_thread(
    session: AsyncSession, *, user_id: UUID, company_id: str, news_id: UUID
) -> NewsChatThread:
    thread = NewsChatThread(user_id=user_id, company_id=company_id, news_id=news_id)
    session.add(thread)
    await session.commit()
    await session.refresh(thread)
    return thread


async def get_owned_thread(
    session: AsyncSession,
    *,
    thread_id: UUID,
    user_id: UUID,
    company_id: str,
    news_id: UUID,
) -> NewsChatThread:
    thread = await session.scalar(
        select(NewsChatThread).where(
            NewsChatThread.id == thread_id,
            NewsChatThread.user_id == user_id,
            NewsChatThread.company_id == company_id,
            NewsChatThread.news_id == news_id,
        )
    )
    if thread is None:
        raise ChatThreadNotFound
    return thread


async def list_messages(
    session: AsyncSession, thread_id: UUID, *, limit: int, before: UUID | None = None
) -> tuple[list[NewsChatMessage], bool]:
    conditions = [NewsChatMessage.thread_id == thread_id]
    if before is not None:
        cursor = await session.get(NewsChatMessage, before)
        if cursor is None or cursor.thread_id != thread_id:
            raise ChatThreadNotFound
        conditions.append(
            or_(
                NewsChatMessage.created_at < cursor.created_at,
                and_(
                    NewsChatMessage.created_at == cursor.created_at,
                    NewsChatMessage.id < cursor.id,
                ),
            )
        )
    rows = list(
        await session.scalars(
            select(NewsChatMessage)
            .where(*conditions)
            .order_by(NewsChatMessage.created_at.desc(), NewsChatMessage.id.desc())
            .limit(limit + 1)
        )
    )
    return list(reversed(rows[:limit])), len(rows) > limit


async def reserve_exchange(
    session: AsyncSession,
    *,
    thread_id: UUID,
    user_id: UUID,
    client_message_id: UUID,
    question: str,
    lease_seconds: int,
) -> tuple[NewsChatMessage, NewsChatMessage, UUID | None, bool]:
    """Persist an idempotent user message and acquire its single answer lease."""

    now = datetime.now(UTC)
    user_message_id = uuid4()
    inserted_id = await session.scalar(
        insert(NewsChatMessage)
        .values(
            id=user_message_id,
            thread_id=thread_id,
            client_message_id=client_message_id,
            user_id=user_id,
            role="user",
            status="ready",
            content=question,
            sources=[],
            warnings=[],
            created_at=now,
        )
        .on_conflict_do_nothing(index_elements=["thread_id", "client_message_id"])
        .returning(NewsChatMessage.id)
    )
    reused = inserted_id is None
    if reused:
        user_message = await session.scalar(
            select(NewsChatMessage).where(
                NewsChatMessage.thread_id == thread_id,
                NewsChatMessage.client_message_id == client_message_id,
            )
        )
        if user_message is None or user_message.content != question:
            await session.rollback()
            raise ValueError("client_message_id already belongs to a different question")
    else:
        user_message = await session.get(NewsChatMessage, inserted_id)

    token = uuid4()
    answer_id = uuid4()
    lease_until = now + timedelta(seconds=lease_seconds + 30)
    await session.execute(
        insert(NewsChatMessage)
        .values(
            id=answer_id,
            thread_id=thread_id,
            reply_to_id=user_message.id,
            role="assistant",
            status="generating",
            lease_token=token,
            lease_until=lease_until,
            content="",
            sources=[],
            warnings=[],
            created_at=now,
        )
        .on_conflict_do_update(
            index_elements=["reply_to_id"],
            set_={
                "status": "generating",
                "lease_token": token,
                "lease_until": lease_until,
                "content": "",
                "answer_kind": None,
                "sources": [],
                "warnings": [],
            },
            where=or_(
                NewsChatMessage.status == "failed",
                (NewsChatMessage.status == "generating")
                & (NewsChatMessage.lease_until < now),
            ),
        )
    )
    assistant = await session.scalar(
        select(NewsChatMessage)
        .where(NewsChatMessage.reply_to_id == user_message.id)
        .execution_options(populate_existing=True)
    )
    await session.execute(
        update(NewsChatThread)
        .where(NewsChatThread.id == thread_id)
        .values(updated_at=now)
    )
    await session.commit()
    return user_message, assistant, token if assistant.lease_token == token else None, reused


async def complete_answer(
    session: AsyncSession,
    *,
    message_id: UUID,
    token: UUID,
    content: str,
    answer_kind: str,
    sources: list[dict],
    warnings: list[str],
) -> NewsChatMessage | None:
    saved = await session.scalar(
        update(NewsChatMessage)
        .where(
            NewsChatMessage.id == message_id,
            NewsChatMessage.status == "generating",
            NewsChatMessage.lease_token == token,
        )
        .values(
            status="ready",
            lease_token=None,
            lease_until=None,
            content=content,
            answer_kind=answer_kind,
            sources=sources,
            warnings=warnings,
        )
        .returning(NewsChatMessage.id)
    )
    await session.commit()
    return await session.get(NewsChatMessage, saved) if saved else None


async def fail_answer(
    session: AsyncSession, *, message_id: UUID, token: UUID, warning: str
) -> None:
    await session.execute(
        update(NewsChatMessage)
        .where(
            NewsChatMessage.id == message_id,
            NewsChatMessage.status == "generating",
            NewsChatMessage.lease_token == token,
        )
        .values(
            status="failed",
            lease_token=None,
            lease_until=None,
            content="",
            warnings=[warning],
        )
    )
    await session.commit()
