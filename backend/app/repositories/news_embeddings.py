"""Durable preprocessing queue and read model for AI-enriched news."""

import asyncio
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from sqlalchemy import and_, func, literal, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.news import NewsArticle
from app.models.news_embeddings import EMBEDDING_DIM, NewsEmbedding
from app.services.llm.config import ServicesSettings, get_services_settings
from app.services.llm.embeddings import EmbeddingsClient, get_embedder
from app.services.llm.matching import score_relevance
from app.services.llm.news_analysis import NEWS_PROMPT_VERSION, process_news_text
from app.services.llm.provider import (
    LLMBudgetExceededError,
    LLMProvider,
    MockProvider,
    get_llm_provider,
)
from app.services.llm.schemas import (
    CompanyFields,
    CompanyProfile,
    NewsAnalysis,
    NewsEvent,
    ProcessedNews,
    RelevanceScore,
)

logger = logging.getLogger("app.repositories.news_embeddings")
DEFAULT_RANK_CANDIDATES = 500
READY = "ready"


class ArticleNotFoundError(LookupError):
    pass


class NewsLeaseLostError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessOutcome:
    processed: dict[UUID, ProcessedNews] = field(default_factory=dict)
    missing: list[UUID] = field(default_factory=list)
    skipped: list[UUID] = field(default_factory=list)
    failed: dict[UUID, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NewsClaim:
    article_id: UUID
    lease_token: UUID
    text: str
    url: str
    source_content_hash: str
    input_fingerprint: str
    attempt: int


def _to_uuid(value: UUID | str) -> UUID:
    if isinstance(value, UUID):
        return value
    text = str(value).strip()
    try:
        return UUID(text)
    except ValueError as exc:
        raise ArticleNotFoundError(text) from exc


def _news_text(article: NewsArticle) -> str:
    return "\n\n".join(filter(None, [article.title, article.content_text or article.summary]))


def _article_url(article: NewsArticle) -> str:
    return getattr(article, "canonical_url", None) or article.original_url


def _source_content_hash(article: NewsArticle) -> str:
    value = getattr(article, "content_hash", None)
    return value or hashlib.sha256(_news_text(article).encode("utf-8")).hexdigest()


def active_embedding_model_version(settings: ServicesSettings) -> str:
    """Return the configured BGE identity without making a network call."""
    return f"{settings.embed_model}:{settings.embed_dim}d"


def _safe_api_identity(base_url: str | None) -> str:
    if not base_url:
        return "unconfigured"
    parsed = urlsplit(base_url)
    host = parsed.hostname or "unknown"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme or 'http'}://{host}{path}"


def llm_model_version(settings: ServicesSettings, llm: LLMProvider | None = None) -> str:
    """Identity used for cache isolation; credentials are intentionally excluded."""
    if isinstance(llm, MockProvider) or (llm is None and settings.llm_mock):
        kind = "mock"
    elif llm is not None and llm.__class__.__name__ != "OpenAICompatProvider":
        kind = f"custom:{llm.__class__.__module__}.{llm.__class__.__qualname__}"
    else:
        kind = f"openai-compatible:{_safe_api_identity(settings.llm_api_base)}"
    return f"{kind}:{settings.model_for('news_analysis')}"


def news_input_fingerprint(
    article: NewsArticle,
    settings: ServicesSettings,
    *,
    llm: LLMProvider | None = None,
    embedding_model_version: str | None = None,
) -> str:
    """Hash every input whose change must invalidate the preprocessed result."""
    text = _news_text(article)
    payload = {
        "source_content_hash": _source_content_hash(article),
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt_version": NEWS_PROMPT_VERSION,
        "llm_model_version": llm_model_version(settings, llm),
        "embedding_model_version": embedding_model_version
        or active_embedding_model_version(settings),
        "news_max_chars": settings.news_max_chars,
        "llm_news_temperature": settings.llm_news_temperature,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _embedding_is_current(
    embedding: NewsEmbedding,
    article: NewsArticle,
    settings: ServicesSettings,
    *,
    llm: LLMProvider | None = None,
    embedding_model_version: str | None = None,
) -> bool:
    embed_version = embedding_model_version or active_embedding_model_version(settings)
    expected = news_input_fingerprint(
        article, settings, llm=llm, embedding_model_version=embed_version
    )
    expected_llm = llm_model_version(settings, llm)
    return (
        getattr(embedding, "status", READY) == READY
        and getattr(embedding, "summary", "legacy") is not None
        and getattr(embedding, "vector", "legacy") is not None
        and getattr(embedding, "source_content_hash", _source_content_hash(article))
        == _source_content_hash(article)
        and getattr(embedding, "prompt_version", NEWS_PROMPT_VERSION) == NEWS_PROMPT_VERSION
        and getattr(embedding, "llm_model_version", expected_llm) == expected_llm
        and embedding.embedding_model_version == embed_version
        and getattr(embedding, "input_fingerprint", expected) == expected
    )


def _to_analysis(embedding: NewsEmbedding) -> NewsAnalysis:
    if embedding.summary is None:
        raise ValueError("AI summary is not ready")
    return NewsAnalysis(**(embedding.analysis or {}), summary=embedding.summary)


def _to_processed(embedding: NewsEmbedding, article: NewsArticle) -> ProcessedNews:
    if embedding.summary is None or embedding.vector is None:
        raise ValueError("AI news payload is not ready")
    return ProcessedNews(
        url=_article_url(article),
        text=_news_text(article),
        summary=embedding.summary,
        markup=_to_analysis(embedding),
        vector=list(embedding.vector),
        embedding_model_version=embedding.embedding_model_version,
    )


def _to_event(embedding: NewsEmbedding, article: NewsArticle) -> NewsEvent:
    if embedding.summary is None:
        raise ValueError("AI summary is not ready")
    return NewsEvent(
        url=_article_url(article),
        title=article.title,
        text=_news_text(article),
        summary=embedding.summary,
        markup=_to_analysis(embedding),
        published_at=(
            article.published_at.isoformat() if getattr(article, "published_at", None) else None
        ),
    )


async def load_ready_news_embeddings(
    session: AsyncSession,
    articles: Sequence[NewsArticle],
    settings: ServicesSettings | None = None,
) -> dict[UUID, NewsEmbedding]:
    """Bulk, read-only readiness check. It never contacts LLM or Ollama."""
    if not articles:
        return {}
    settings = settings or get_services_settings()
    by_id = {article.id: article for article in articles}
    rows = await session.scalars(
        select(NewsEmbedding).where(
            NewsEmbedding.article_id.in_(by_id),
            NewsEmbedding.status == READY,
            NewsEmbedding.prompt_version == NEWS_PROMPT_VERSION,
            NewsEmbedding.llm_model_version == llm_model_version(settings),
            NewsEmbedding.embedding_model_version == active_embedding_model_version(settings),
        )
    )
    return {
        row.article_id: row
        for row in rows
        if _embedding_is_current(row, by_id[row.article_id], settings)
    }


async def fetch_news_records(
    session: AsyncSession,
    article_ids: Sequence[UUID | str],
    *,
    settings: ServicesSettings | None = None,
) -> list[ProcessedNews]:
    ids = list(dict.fromkeys(_to_uuid(value) for value in article_ids))
    if not ids:
        return []
    articles = list(await session.scalars(select(NewsArticle).where(NewsArticle.id.in_(ids))))
    ready = await load_ready_news_embeddings(session, articles, settings)
    by_id = {article.id: article for article in articles}
    return [_to_processed(ready[value], by_id[value]) for value in ids if value in ready]


async def fetch_news_events(
    session: AsyncSession,
    article_ids: Sequence[UUID | str],
    *,
    settings: ServicesSettings | None = None,
) -> list[NewsEvent]:
    ids = list(dict.fromkeys(_to_uuid(value) for value in article_ids))
    if not ids:
        return []
    articles = list(await session.scalars(select(NewsArticle).where(NewsArticle.id.in_(ids))))
    ready = await load_ready_news_embeddings(session, articles, settings)
    by_id = {article.id: article for article in articles}
    return [_to_event(ready[value], by_id[value]) for value in ids if value in ready]


async def fetch_news_event(
    session: AsyncSession,
    article: NewsArticle,
    *,
    allow_rss_fallback: bool = True,
    settings: ServicesSettings | None = None,
) -> tuple[NewsEvent, list[str], str]:
    """Return current AI input or an explicitly labelled RSS fallback for detail/chat."""
    settings = settings or get_services_settings()
    fingerprint = news_input_fingerprint(article, settings)
    ready = await load_ready_news_embeddings(session, [article], settings)
    embedding = ready.get(article.id)
    if embedding is not None:
        return _to_event(embedding, article), [], fingerprint
    if not allow_rss_fallback:
        raise LookupError("AI-анализ новости ещё не готов")
    event = NewsEvent(
        url=_article_url(article),
        title=article.title,
        text=_news_text(article)[: settings.news_max_chars],
        summary=article.summary or article.title,
        markup=None,
        published_at=(
            article.published_at.isoformat() if getattr(article, "published_at", None) else None
        ),
    )
    return event, ["AI-анализ ещё не готов; использован исходный RSS-текст."], fingerprint


async def rank_relevant_for_company(
    session: AsyncSession,
    profile: CompanyProfile,
    fields: CompanyFields,
    *,
    article_ids: Sequence[UUID | str] | None = None,
    settings: ServicesSettings | None = None,
    min_score: float | None = None,
    limit: int | None = None,
    since: datetime | None = None,
    candidates: int | None = DEFAULT_RANK_CANDIDATES,
) -> list[RelevanceScore]:
    settings = settings or get_services_settings()
    embed_version = active_embedding_model_version(settings)
    if profile.embedding_model_version != embed_version:
        return []
    query = (
        select(NewsEmbedding, NewsArticle)
        .join(NewsArticle, NewsArticle.id == NewsEmbedding.article_id)
        .where(
            NewsEmbedding.status == READY,
            NewsEmbedding.source_content_hash == NewsArticle.content_hash,
            NewsEmbedding.prompt_version == NEWS_PROMPT_VERSION,
            NewsEmbedding.llm_model_version == llm_model_version(settings),
            NewsEmbedding.embedding_model_version == embed_version,
        )
    )
    if article_ids is not None:
        ids = [_to_uuid(value) for value in article_ids]
        if not ids:
            return []
        query = query.where(NewsEmbedding.article_id.in_(ids))
    if since is not None:
        query = query.where(NewsArticle.first_seen_at >= since)
    main_vector = profile.vectors.get("main", [])
    if candidates is not None and len(main_vector) == EMBEDDING_DIM:
        query = query.order_by(NewsEmbedding.vector.cosine_distance(main_vector)).limit(
            max(candidates, limit or 0)
        )
    rows = await session.execute(query)
    pairs = [
        (str(embedding.article_id), _to_processed(embedding, article))
        for embedding, article in rows.all()
        if _embedding_is_current(embedding, article, settings)
    ]
    scores = [
        score_relevance(profile, fields, item, settings=settings, news_id=news_id)
        for news_id, item in pairs
    ]
    scores.sort(key=lambda item: item.score, reverse=True)
    if min_score is not None:
        scores = [item for item in scores if item.score >= min_score]
    return scores[:limit] if limit is not None else scores


async def process_and_store_news(
    session: AsyncSession,
    article_ids: Sequence[UUID | str],
    *,
    llm: LLMProvider | None = None,
    embedder: EmbeddingsClient | None = None,
    settings: ServicesSettings | None = None,
    force: bool = False,
    lease_token: UUID | None = None,
) -> ProcessOutcome:
    """Process explicit ids, isolating failures and invalidating every changed input."""
    ids = [_to_uuid(value) for value in article_ids]
    if not ids:
        return ProcessOutcome()
    settings = settings or get_services_settings()
    if settings.embed_dim != EMBEDDING_DIM:
        raise ValueError(f"AI_EMBED_DIM={settings.embed_dim}, expected {EMBEDDING_DIM}")
    llm = llm or get_llm_provider(settings)
    embedder = embedder or get_embedder()
    unique_ids = list(dict.fromkeys(ids))
    articles = list(
        await session.scalars(select(NewsArticle).where(NewsArticle.id.in_(unique_ids)))
    )
    found = {article.id: article for article in articles}
    missing = [value for value in unique_ids if value not in found]
    stored = {
        row.article_id: row
        for row in await session.scalars(
            select(NewsEmbedding).where(NewsEmbedding.article_id.in_(unique_ids))
        )
    }
    outcome = ProcessOutcome(missing=missing)
    pending = [
        value
        for value in unique_ids
        if value in found
        and (
            force
            or value not in stored
            or not _embedding_is_current(
                stored[value], found[value], settings, llm=llm,
                embedding_model_version=embedder.model_version
            )
        )
    ]
    outcome.skipped.extend(value for value in found if value not in pending)
    for position, article_id in enumerate(pending):
        article = found[article_id]
        try:
            processed = await asyncio.to_thread(
                process_news_text, _news_text(article), article.original_url,
                llm=llm, embedder=embedder, settings=settings
            )
            if len(processed.vector) != EMBEDDING_DIM:
                raise ValueError(f"Размерность вектора {len(processed.vector)} != {EMBEDDING_DIM}")
        except LLMBudgetExceededError as exc:
            logger.warning("News processing stopped because token budget is exhausted")
            for left in pending[position:]:
                outcome.failed[left] = str(exc)
            break
        except Exception as exc:
            logger.warning(
                "News processing failed article=%s type=%s", article_id, type(exc).__name__
            )
            outcome.failed[article_id] = str(exc)
            continue
        await _upsert_embedding(
            session, article, processed, settings=settings, llm=llm, lease_token=lease_token
        )
        await session.commit()
        outcome.processed[article_id] = processed
    return outcome


async def _upsert_embedding(
    session: AsyncSession,
    article: NewsArticle,
    processed: ProcessedNews,
    *,
    settings: ServicesSettings,
    llm: LLMProvider,
    lease_token: UUID | None = None,
) -> None:
    analysis: dict[str, Any] = {
        key: value for key, value in processed.markup.model_dump(mode="json").items()
        if key != "summary"
    }
    values = {
        "article_id": article.id,
        "summary": processed.summary,
        "vector": processed.vector,
        "analysis": analysis,
        "source_content_hash": _source_content_hash(article),
        "input_fingerprint": news_input_fingerprint(
            article, settings, llm=llm,
            embedding_model_version=processed.embedding_model_version
        ),
        "llm_model_version": llm_model_version(settings, llm),
        "prompt_version": NEWS_PROMPT_VERSION,
        "embedding_model_version": processed.embedding_model_version,
        "status": READY,
        "attempts": 0,
        "next_attempt_at": None,
        "last_error": None,
        "lease_token": None,
        "lease_until": None,
        "updated_at": func.now(),
    }
    statement = insert(NewsEmbedding).values(**values)
    conflict = statement.on_conflict_do_update(
        index_elements=["article_id"],
        set_={key: getattr(statement.excluded, key) for key in values if key != "article_id"},
        where=(NewsEmbedding.lease_token == lease_token) if lease_token else None,
    )
    if lease_token:
        result = await session.execute(conflict.returning(NewsEmbedding.article_id))
        if result.scalar_one_or_none() is None:
            raise NewsLeaseLostError(str(article.id))
    else:
        await session.execute(conflict)


async def seed_news_jobs(
    factory: async_sessionmaker,
    batch_size: int,
    *,
    article_ids: Sequence[UUID] | None = None,
) -> int:
    """Gradually attach queue rows to accumulated RSS articles."""
    async with factory() as session:
        missing_query = (
            select(NewsArticle.id)
            .outerjoin(NewsEmbedding, NewsEmbedding.article_id == NewsArticle.id)
            .where(NewsEmbedding.article_id.is_(None))
        )
        if article_ids is not None:
            missing_query = missing_query.where(NewsArticle.id.in_(article_ids))
        missing_ids = (
            missing_query.order_by(NewsArticle.first_seen_at, NewsArticle.id)
            .limit(batch_size)
            .subquery()
        )
        values = select(
            missing_ids.c.id,
            func.jsonb_build_object(),
            literal(""),
            literal(""),
            literal(""),
            literal(""),
            literal(""),
            literal("pending"),
            literal(0),
        )
        statement = insert(NewsEmbedding).from_select(
            ["article_id", "analysis", "source_content_hash", "input_fingerprint",
             "llm_model_version", "prompt_version", "embedding_model_version", "status",
             "attempts"],
            values,
        ).on_conflict_do_nothing(index_elements=["article_id"])
        result = await session.execute(statement)
        await session.commit()
        return result.rowcount or 0


async def claim_news_batch(
    factory: async_sessionmaker,
    settings: ServicesSettings,
    *,
    llm: LLMProvider,
    embedder: EmbeddingsClient,
    article_ids: Sequence[UUID] | None = None,
) -> list[NewsClaim]:
    await seed_news_jobs(
        factory, settings.news_worker_batch_size, article_ids=article_ids
    )
    now = datetime.now(UTC)
    embed_version = embedder.model_version
    llm_version = llm_model_version(settings, llm)
    stale = or_(
        NewsEmbedding.source_content_hash != NewsArticle.content_hash,
        NewsEmbedding.prompt_version != NEWS_PROMPT_VERSION,
        NewsEmbedding.llm_model_version != llm_version,
        NewsEmbedding.embedding_model_version != embed_version,
    )
    due = or_(
        NewsEmbedding.status == "pending",
        and_(
            NewsEmbedding.status.in_(["retry", "blocked_budget"]),
            or_(NewsEmbedding.next_attempt_at.is_(None), NewsEmbedding.next_attempt_at <= now),
        ),
        and_(
            NewsEmbedding.status == "processing",
            or_(NewsEmbedding.lease_until.is_(None), NewsEmbedding.lease_until < now),
        ),
    )
    active_lease = and_(
        NewsEmbedding.status == "processing",
        NewsEmbedding.lease_until.is_not(None),
        NewsEmbedding.lease_until >= now,
    )
    async with factory() as session:
        query = (
            select(NewsEmbedding, NewsArticle)
                .join(NewsArticle, NewsArticle.id == NewsEmbedding.article_id)
                .where(or_(due, and_(stale, ~active_lease)))
        )
        if article_ids is not None:
            query = query.where(NewsEmbedding.article_id.in_(article_ids))
        rows = (
            await session.execute(
                query.order_by(NewsArticle.first_seen_at, NewsArticle.id)
                .limit(min(settings.news_worker_batch_size, settings.news_worker_concurrency))
                .with_for_update(of=NewsEmbedding, skip_locked=True)
            )
        ).all()
        claims: list[NewsClaim] = []
        for embedding, article in rows:
            changed = (
                embedding.source_content_hash != article.content_hash
                or embedding.prompt_version != NEWS_PROMPT_VERSION
                or embedding.llm_model_version != llm_version
                or embedding.embedding_model_version != embed_version
            )
            token = uuid4()
            embedding.status = "processing"
            embedding.attempts = 1 if changed else embedding.attempts + 1
            # Persist the exact claimed snapshot before releasing the row lock.
            # This keeps a second worker from treating a live claim as stale.
            embedding.source_content_hash = article.content_hash
            embedding.input_fingerprint = news_input_fingerprint(
                article, settings, llm=llm, embedding_model_version=embed_version
            )
            embedding.llm_model_version = llm_version
            embedding.prompt_version = NEWS_PROMPT_VERSION
            embedding.embedding_model_version = embed_version
            embedding.next_attempt_at = None
            embedding.last_error = None
            embedding.lease_token = token
            embedding.lease_until = now + timedelta(seconds=settings.news_worker_lease_seconds)
            claims.append(
                NewsClaim(
                    article_id=article.id,
                    lease_token=token,
                    text=_news_text(article),
                    url=article.original_url,
                    source_content_hash=article.content_hash,
                    input_fingerprint=embedding.input_fingerprint,
                    attempt=embedding.attempts,
                )
            )
        await session.commit()
        return claims


async def renew_news_lease(
    factory: async_sessionmaker, claim: NewsClaim, lease_seconds: int
) -> None:
    async with factory() as session:
        result = await session.execute(
            update(NewsEmbedding)
            .where(
                NewsEmbedding.article_id == claim.article_id,
                NewsEmbedding.lease_token == claim.lease_token,
                NewsEmbedding.status == "processing",
            )
            .values(lease_until=datetime.now(UTC) + timedelta(seconds=lease_seconds))
        )
        await session.commit()
        if result.rowcount != 1:
            raise NewsLeaseLostError(str(claim.article_id))


async def store_news_claim(
    factory: async_sessionmaker,
    claim: NewsClaim,
    processed: ProcessedNews,
    settings: ServicesSettings,
    llm: LLMProvider,
) -> None:
    async with factory() as session:
        article = await session.get(NewsArticle, claim.article_id)
        if article is None:
            raise ArticleNotFoundError(str(claim.article_id))
        if news_input_fingerprint(
            article, settings, llm=llm,
            embedding_model_version=processed.embedding_model_version
        ) != claim.input_fingerprint:
            await session.rollback()
            await mark_news_claim_failed(
                factory, claim, "Источник изменился во время обработки", settings
            )
            return
        await _upsert_embedding(
            session, article, processed, settings=settings, llm=llm,
            lease_token=claim.lease_token
        )
        await session.commit()


def _retry_delay(settings: ServicesSettings, attempt: int) -> int:
    return min(settings.news_worker_retry_seconds * 2 ** max(attempt - 1, 0), 3600)


async def mark_news_claim_failed(
    factory: async_sessionmaker,
    claim: NewsClaim,
    error: str,
    settings: ServicesSettings,
    *,
    budget_blocked: bool = False,
) -> None:
    now = datetime.now(UTC)
    if budget_blocked:
        tomorrow = (now + timedelta(days=1)).date()
        next_attempt = datetime.combine(tomorrow, datetime.min.time(), tzinfo=UTC) + timedelta(
            minutes=1
        )
        status = "blocked_budget"
    elif claim.attempt >= settings.news_worker_max_attempts:
        next_attempt, status = None, "failed"
    else:
        next_attempt = now + timedelta(seconds=_retry_delay(settings, claim.attempt))
        status = "retry"
    async with factory() as session:
        result = await session.execute(
            update(NewsEmbedding)
            .where(
                NewsEmbedding.article_id == claim.article_id,
                NewsEmbedding.lease_token == claim.lease_token,
            )
            .values(
                status=status,
                next_attempt_at=next_attempt,
                last_error=error[:1000],
                lease_token=None,
                lease_until=None,
                updated_at=func.now(),
            )
        )
        await session.commit()
        if result.rowcount != 1:
            raise NewsLeaseLostError(str(claim.article_id))
