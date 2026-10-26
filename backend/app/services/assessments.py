"""Short company news assessment: risks, opportunities and bank analytics in 2-4 sentences each.

One LLM call per (company, news) instead of the Analyst -> Verifier -> Revise loop, so the feed
can show it quickly. The strict, source-checked flow stays in POST .../interpretation.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.company import Company
from app.models.news import NewsArticle
from app.repositories.assessments import (
    AssessmentClaim,
    claim_assessments,
    complete_assessment,
    fail_assessment,
)
from app.repositories.news_embeddings import (
    fetch_news_event,
    llm_model_version,
    news_input_fingerprint,
)
from app.schemas.company_news import InterpretationSource
from app.services.analytics.selection import effective_bank_settings
from app.services.company_news import (
    DeadlineClient,
    bank_query,
    company_context,
    company_input_fingerprint,
    fingerprint_of,
    require_llm,
    retrieve_bank_passages,
)
from app.services.llm.config import ServicesSettings
from app.services.llm.provider import (
    LLMBudgetExceededError,
    LLMProvider,
    MockProvider,
    OpenAICompatProvider,
    complete_json,
    messages,
)
from app.services.llm.schemas import AssessmentDraft, BankPassage, CompanyContext, NewsEvent
from app.services.llm.text import shorten_text

logger = logging.getLogger("app.services.assessments")

ASSESSMENT_PROMPT_VERSION = 3
MAX_SENTENCES = 2
MAX_BLOCK_CHARS = 280
NEWS_TEXT_CHARS = 2000
PASSAGE_CHARS = 900

ASSESSMENT_SYSTEM_PROMPT = (
    "Ты — финансовый аналитик Альфа-Банка. По новости и профилю клиента-компании дай "
    "ОЧЕНЬ КРАТКУЮ оценку для ленты. Верни строго JSON без markdown:\n"
    "{\n"
    '  "brief_impact": "1-2 коротких предложения: почему новость важна именно этой компании",\n'
    '  "risks": "1-2 коротких предложения: чем новость может навредить именно этой компании",\n'
    '  "opportunities": "1-2 коротких предложения: что новость даёт именно этой компании",\n'
    '  "bank_analytics": "1-2 коротких предложения: позиция аналитиков Альфа-Банка по теме '
    'новости или null",\n'
    '  "bank_refs": [1]\n'
    "}\n"
    "Правила: пиши по-русски, максимум 1-2 предложения и до 250 символов на блок, "
    "по делу, без вступлений и списков. "
    "Опирайся на профиль компании (отрасль, регион, продукты банка, масштаб) и текст новости; "
    "не выдумывай цифры и факты. Если значимых рисков (или возможностей) для этой компании "
    "нет — верни null в соответствующем поле, ничего не пиши про «рисков нет». "
    "bank_analytics заполняй ТОЛЬКО по блоку «Аналитика Альфа-Банка» и только если фрагменты "
    "относятся к теме новости; в bank_refs укажи номера использованных фрагментов [B#]. "
    "Если подходящих фрагментов нет — bank_analytics: null, bank_refs: []. "
    "Не приписывай банку выводы, которых нет во фрагментах."
)


def assessment_fingerprint(
    company_fp: str, article: NewsArticle, settings: ServicesSettings
) -> str:
    """Hash of every input whose change must regenerate the assessment (no network calls)."""

    return fingerprint_of(
        {
            "version": ASSESSMENT_PROMPT_VERSION,
            "company": company_fp,
            "news": news_input_fingerprint(article, settings),
            "llm": llm_model_version(settings),
            "temperature": settings.llm_temperature,
            "bank_documents": sorted(map(str, settings.bank_document_ids)),
            "bank_passages": settings.assessment_bank_passages,
            "prompt": ASSESSMENT_SYSTEM_PROMPT,
        }
    )


def llm_is_configured(settings: ServicesSettings) -> bool:
    return bool(settings.llm_mock or settings.llm_api_base)


def build_messages(
    context: CompanyContext, news: NewsEvent, passages: list[BankPassage]
) -> list[dict[str, str]]:
    lines = ["=== ПРОФИЛЬ КОМПАНИИ ===", context.profile_text, "", "=== НОВОСТЬ ==="]
    if news.title:
        lines.append(f"Заголовок: {news.title}")
    if news.published_at:
        lines.append(f"Дата: {news.published_at}")
    lines.append(f"Суммари: {news.summary}")
    if news.text and news.text != news.summary:
        lines.append(f"Текст:\n{news.text[:NEWS_TEXT_CHARS]}")
    lines += ["", "=== АНАЛИТИКА АЛЬФА-БАНКА ==="]
    if passages:
        for number, passage in enumerate(passages, start=1):
            published = f", {passage.published_at}" if passage.published_at else ""
            lines.append(f"[B{number}] {passage.title}{published}\n{passage.text[:PASSAGE_CHARS]}")
    else:
        lines.append("(релевантных материалов нет)")
    lines.append("\nДай краткую оценку.")
    return messages(ASSESSMENT_SYSTEM_PROMPT, "\n".join(lines))


def shorten(text: str) -> str:
    """At most MAX_SENTENCES sentences and MAX_BLOCK_CHARS characters."""

    return shorten_text(text, sentences=MAX_SENTENCES, chars=MAX_BLOCK_CHARS)


def finalize(
    draft: AssessmentDraft,
    passages: list[BankPassage],
    sources: list[InterpretationSource],
) -> dict:
    """Keep bank analytics only when it cites real, retrieved passages."""

    refs = sorted({ref for ref in draft.bank_refs if 1 <= ref <= len(passages)})
    bank_text = shorten(draft.bank_analytics) if draft.bank_analytics else None
    if not bank_text or not refs or len(passages) != len(sources):
        bank_text, refs = None, []
    return {
        "brief_impact": shorten(draft.brief_impact) if draft.brief_impact else None,
        "risks": shorten(draft.risks) if draft.risks else None,
        "opportunities": shorten(draft.opportunities) if draft.opportunities else None,
        "bank_analytics": bank_text,
        "bank_sources": [sources[ref - 1].model_dump(mode="json") for ref in refs],
        "bank_passages": [passages[ref - 1].model_dump(mode="json") for ref in refs],
    }


def generate_assessment(
    context: CompanyContext,
    news: NewsEvent,
    passages: list[BankPassage],
    settings: ServicesSettings,
    *,
    llm: LLMProvider | None = None,
    deadline: float | None = None,
) -> AssessmentDraft:
    require_llm(settings)
    prompt = build_messages(context, news, passages)
    if llm is not None or settings.llm_mock:
        draft, _ = complete_json(
            "news_assessment",
            AssessmentDraft,
            prompt,
            provider=llm or MockProvider(),
            settings=settings,
        )
        return draft
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    with DeadlineClient(
        deadline=deadline or time.monotonic() + settings.assessment_lease_seconds - 15,
        per_call_timeout=settings.llm_timeout_seconds,
        base_url=settings.llm_api_base.rstrip("/"),
        headers={"Authorization": f"Bearer {key}"} if key else {},
    ) as client:
        provider = OpenAICompatProvider(
            settings.llm_api_base,
            key,
            settings.llm_timeout_seconds,
            client=client,
            reasoning_budget=settings.llm_reasoning_budget,
        )
        draft, _ = complete_json(
            "news_assessment", AssessmentDraft, prompt, provider=provider, settings=settings
        )
    return draft


async def build_assessment(
    session: AsyncSession,
    company: Company,
    article: NewsArticle,
    settings: ServicesSettings,
    *,
    llm: LLMProvider | None = None,
    deadline: float | None = None,
) -> tuple[str, dict]:
    """Fingerprint and stored payload for one (company, news) pair."""

    settings = await effective_bank_settings(session, settings)
    context = company_context(company)
    news, warnings, _ = await fetch_news_event(
        session, article, allow_rss_fallback=True, settings=settings
    )
    passages, sources, bank_warnings = await retrieve_bank_passages(
        session,
        query=bank_query(article, company),
        settings=settings,
        as_of=(article.published_at or datetime.now(UTC)).date(),
        limit=settings.assessment_bank_passages,
    )
    fingerprint = assessment_fingerprint(company_input_fingerprint(context), article, settings)
    # The DB transaction must not stay open during the LLM call.
    await session.commit()
    draft = await asyncio.to_thread(
        generate_assessment, context, news, passages, settings, llm=llm, deadline=deadline
    )
    payload = finalize(draft, passages, sources)
    payload.update(
        {
            "model": settings.llm_model,
            "is_demo": settings.llm_mock,
            "generated_at": datetime.now(UTC).isoformat(),
            "warnings": list(dict.fromkeys([*warnings, *bank_warnings])),
        }
    )
    return fingerprint, payload


class AssessmentWorker:
    """Polls the news_assessments queue and fills it with short LLM assessments."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: ServicesSettings,
        *,
        llm: LLMProvider | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._llm = llm
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._pause_seconds = 0.0

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="news-assessment-worker")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except TimeoutError:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            self._pause_seconds = 0.0
            try:
                processed = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("News assessment polling iteration failed")
                processed = 0
                self._pause_seconds = self._settings.assessment_retry_seconds
            delay = self._pause_seconds or (
                0 if processed else self._settings.assessment_poll_seconds
            )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def run_once(self) -> int:
        settings = self._settings
        if not settings.assessment_enabled or not llm_is_configured(settings):
            self._pause_seconds = settings.news_worker_config_retry_seconds
            return 0
        async with self._session_factory() as session:
            claims = await claim_assessments(
                session,
                limit=settings.assessment_batch_size,
                lease_seconds=settings.assessment_lease_seconds,
            )
        semaphore = asyncio.Semaphore(settings.assessment_concurrency)

        async def bounded(claim: AssessmentClaim) -> None:
            async with semaphore:
                await self._process(claim)

        async with asyncio.TaskGroup() as group:
            for claim in claims:
                group.create_task(bounded(claim), name=f"news-assessment-{claim.news_id}")
        return len(claims)

    async def _process(self, claim: AssessmentClaim) -> None:
        settings = self._settings
        try:
            async with self._session_factory() as session:
                company = await session.get(Company, claim.company_id)
                article = await session.get(NewsArticle, claim.news_id)
                if company is None or article is None:
                    await fail_assessment(
                        session,
                        claim,
                        error="company or news removed",
                        retry_seconds=settings.assessment_retry_seconds,
                        max_attempts=1,
                    )
                    return
                fingerprint, payload = await build_assessment(
                    session, company, article, settings, llm=self._llm
                )
            async with self._session_factory() as session:
                saved = await complete_assessment(
                    session,
                    claim,
                    fingerprint=fingerprint,
                    result=payload,
                    cache_hours=settings.assessment_cache_hours,
                )
            if not saved:
                logger.warning(
                    "News assessment lease lost news=%s; result discarded", claim.news_id
                )
        except asyncio.CancelledError:
            raise
        except LLMBudgetExceededError as exc:
            await self._fail(claim, str(exc))
            self._pause_seconds = settings.news_worker_config_retry_seconds
        except Exception as exc:
            logger.warning(
                "News assessment failed company=%s news=%s type=%s",
                claim.company_id,
                claim.news_id,
                type(exc).__name__,
            )
            await self._fail(claim, f"{type(exc).__name__}: {exc}")

    async def _fail(self, claim: AssessmentClaim, error: str) -> None:
        async with self._session_factory() as session:
            await fail_assessment(
                session,
                claim,
                error=error,
                retry_seconds=self._settings.assessment_retry_seconds,
                max_attempts=self._settings.assessment_max_attempts,
            )


def assessment_view(payload: dict) -> dict:
    """Feed fields from a stored payload; empty risks/opportunities are None."""

    return {
        "assessment_status": "ready",
        "brief_impact": payload.get("brief_impact"),
        "risks": [payload["risks"]] if payload.get("risks") else None,
        "opportunities": [payload["opportunities"]] if payload.get("opportunities") else None,
        "bank_analytics": payload.get("bank_analytics"),
        "bank_sources": payload.get("bank_sources") or [],
        "generated_at": payload.get("generated_at"),
    }
