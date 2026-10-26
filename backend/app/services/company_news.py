import hashlib
import json
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime
from decimal import Decimal

import httpx2 as httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.company import Company
from app.models.news import NewsArticle
from app.schemas.analytics import SearchRequest
from app.schemas.company_news import InterpretationResponse, InterpretationSource
from app.services.analytics.search import SemanticSearchUnavailable, search_analytics
from app.services.llm.config import ServicesSettings
from app.services.llm.insight import (
    ANALYST_SYSTEM_PROMPT,
    REVISE_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    build_insight,
)
from app.services.llm.provider import MockProvider, OpenAICompatProvider
from app.services.llm.schemas import BankPassage, CompanyContext, InsightResult, NewsEvent

CONTEXT_VERSION = 1
MAX_BANK_PASSAGES = 8
MAX_RETRY_QUERIES = 2
_progress_callback: ContextVar[Callable[[str, dict], None] | None] = ContextVar(
    "interpretation_progress_callback", default=None
)


class LLMUnavailable(RuntimeError):
    pass


@contextmanager
def interpretation_progress(
    callback: Callable[[str, dict], None] | None,
) -> Iterator[None]:
    token = _progress_callback.set(callback)
    try:
        yield
    finally:
        _progress_callback.reset(token)


def emit_interpretation_progress(event: str, data: dict) -> None:
    callback = _progress_callback.get()
    if callback is None:
        return
    try:
        callback(event, data)
    except Exception:
        return


def require_llm(settings: ServicesSettings) -> None:
    if not settings.llm_mock and not settings.llm_api_base:
        raise LLMUnavailable("LLM is not configured; set AI_LLM_API_BASE and AI_LLM_API_KEY")


def company_context(company: Company) -> CompanyContext:
    industry = f"Отрасль (ОКВЭД {company.okved_code})" if company.okved_code else "Отрасль"
    products = [
        code
        for code, value in {
            "rko": company.has_rko,
            "gar": company.has_guarantees,
            "loa": company.has_credit,
            "lea": company.has_leasing,
            "dep": company.has_deposit,
            "zpp": company.has_payroll,
        }.items()
        if value is True
    ]
    fields = {
        "ID компании": company.id,
        "Название": company.name,
        industry: company.okved_name,
        "Тип организации": company.org_type,
        "Форма собственности": company.ownership_form,
        "Возраст": company.age_category,
        "Стадия развития": company.development_stage,
        "Численность сотрудников": company.employees_fns_category,
        "Тип клиента": company.client_type,
        "Продукты банка": ", ".join(products) if products else "неизвестно или не указаны",
        "Выручка": company.annual_revenue,
        "Негативная метка": company.key_owner_flag,
        "Факторинг": company.has_factoring,
        "ВЭД": company.has_fts_data,
        "Страны ВЭД": company.fts_country_name,
        "Регион": company.region,
        "Город": company.city,
        "Группа компаний": company.group_id,
    }

    def display(value):
        if value is None:
            return "неизвестно"
        if isinstance(value, Decimal):
            return format(value, "f")
        if isinstance(value, bool):
            return "да" if value else "нет"
        return str(value)

    text = "\n".join(f"{key}: {display(value)}" for key, value in fields.items())
    text += (
        "\nДоля рынка, размер кредита, валюта и период платежей не предоставлены. "
        "Наличие продукта не определяет его объём. Контрагенты в этот профиль не включены."
    )
    return CompanyContext(
        company_id=company.id,
        profile_text=text,
        okved_code=company.okved_code,
        okved_name=company.okved_name,
        region=company.region,
        city=company.city,
        client_type=company.client_type,
        products=products,
        as_of=company.updated_at.isoformat(),
        evidence_labels=[key for key, value in fields.items() if value is not None],
    )


def fingerprint_of(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def company_input_fingerprint(context: CompanyContext) -> str:
    return fingerprint_of(context.model_dump(mode="json"))


def model_prompt_version(settings: ServicesSettings) -> str:
    """Pure version marker usable by the feed without any AI or retrieval calls."""

    return fingerprint_of(
        {
            "context_version": CONTEXT_VERSION,
            "model": settings.llm_model,
            "provider": settings.llm_api_base,
            "mock": settings.llm_mock,
            "temperature": settings.llm_temperature,
            "cycles": settings.insight_max_cycles,
            "trusted_bank_document_ids": sorted(map(str, settings.bank_document_ids)),
            "prompts": [ANALYST_SYSTEM_PROMPT, VERIFIER_SYSTEM_PROMPT, REVISE_SYSTEM_PROMPT],
        }
    )


def retrieval_fingerprint(passages: list[BankPassage]) -> str:
    return fingerprint_of([passage.model_dump(mode="json") for passage in passages])


def final_cache_key(request_fingerprint: str, passages: list[BankPassage]) -> str:
    return fingerprint_of(
        {"request_fingerprint": request_fingerprint, "retrieval": retrieval_fingerprint(passages)}
    )


def bank_query(article: NewsArticle, company: Company) -> str:
    words = list(
        dict.fromkeys(
            re.findall(r"[а-яёa-z0-9]{3,}", f"{article.title} {company.okved_name or ''}".lower())
        )
    )[:30]
    return " OR ".join(words) or "экономика"


async def retrieve_bank_passages(
    session: AsyncSession,
    *,
    query: str,
    settings: ServicesSettings,
    as_of: date,
    limit: int = 6,
    existing_refs: set[str] | None = None,
) -> tuple[list[BankPassage], list[InterpretationSource], list[str]]:
    """Search only explicitly trusted bank documents, preserving provenance."""

    if not settings.bank_document_ids:
        return [], [], []
    request = SearchRequest(
        query=query[:4000],
        mode="hybrid",
        limit=min(limit, MAX_BANK_PASSAGES),
        document_ids=settings.bank_document_ids,
        as_of=as_of,
    )
    try:
        found = await search_analytics(session, request)
    except SemanticSearchUnavailable:
        found = await search_analytics(session, request.model_copy(update={"mode": "keyword"}))
        found.warnings.append(
            "Семантический поиск недоступен; явно использован только текстовый поиск."
        )
    warnings = list(found.warnings)
    if found.mode_used == "keyword" and not any(
        "текстов" in warning.lower() or "семантичес" in warning.lower() for warning in warnings
    ):
        warnings.append("Семантический поиск недоступен; использован текстовый поиск.")
    passages: list[BankPassage] = []
    sources: list[InterpretationSource] = []
    seen = set(existing_refs or ())
    for hit in found.items:
        evidence_ref = f"bank:{hit.document_id}#{hit.chunk_id}"
        if evidence_ref in seen:
            continue
        seen.add(evidence_ref)
        passages.append(
            BankPassage(
                document_id=str(hit.document_id),
                chunk_id=hit.chunk_id,
                title=hit.title,
                text=hit.text[:4000],
                published_at=hit.published_at.isoformat() if hit.published_at else None,
            )
        )
        sources.append(
            InterpretationSource(
                evidence_ref=evidence_ref,
                title=hit.title,
                source_url=hit.source_url,
                published_at=passages[-1].published_at,
                page_number=hit.page_number,
            )
        )
    return passages, sources, warnings


async def prepare_interpretation(
    session: AsyncSession,
    company: Company,
    article: NewsArticle,
    settings: ServicesSettings,
) -> tuple[
    CompanyContext, NewsEvent, list[BankPassage], list[InterpretationSource], list[str], str
]:
    from app.repositories.news_embeddings import fetch_news_event

    context = company_context(company)
    news, warnings, news_fingerprint = await fetch_news_event(
        session, article, allow_rss_fallback=True, settings=settings
    )
    sources = [
        InterpretationSource(
            evidence_ref=f"news:{news.url}",
            title=article.title,
            source_url=news.url or article.canonical_url,
            published_at=news.published_at,
        )
    ]
    passages, bank_sources, bank_warnings = await retrieve_bank_passages(
        session,
        query=bank_query(article, company),
        settings=settings,
        as_of=(article.published_at or datetime.now(UTC)).date(),
    )
    sources.extend(bank_sources)
    warnings.extend(bank_warnings)
    if not passages:
        warnings.append(
            "Подтверждённые релевантные материалы банка не найдены; позиция банка не сформирована."
        )
    inputs = {
        "version": CONTEXT_VERSION,
        "company_input_fingerprint": company_input_fingerprint(context),
        "news_input_fingerprint": news_fingerprint,
        "news_id": str(article.id),
        "model_prompt_version": model_prompt_version(settings),
    }
    key = fingerprint_of(inputs)
    return context, news, passages, sources, list(dict.fromkeys(warnings)), key


async def retrieve_for_verifier(
    session: AsyncSession,
    *,
    queries: list[str],
    settings: ServicesSettings,
    article: NewsArticle,
    passages: list[BankPassage],
    sources: list[InterpretationSource],
) -> tuple[list[BankPassage], list[InterpretationSource], list[str]]:
    """One bounded evidence expansion cycle requested by the verifier."""

    if not queries or len(passages) >= MAX_BANK_PASSAGES:
        return passages, sources, []
    existing = {
        source.evidence_ref for source in sources if source.evidence_ref.startswith("bank:")
    }
    warnings: list[str] = []
    for query in list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:MAX_RETRY_QUERIES]:
        added, added_sources, added_warnings = await retrieve_bank_passages(
            session,
            query=query[:4000],
            settings=settings,
            as_of=(article.published_at or datetime.now(UTC)).date(),
            limit=MAX_BANK_PASSAGES - len(passages),
            existing_refs=existing,
        )
        passages.extend(added)
        sources.extend(added_sources)
        existing.update(source.evidence_ref for source in added_sources)
        warnings.extend(added_warnings)
        if len(passages) >= MAX_BANK_PASSAGES:
            break
    if queries and not passages:
        warnings.append("Добор подтверждённых банковских источников не дал результатов.")
    return passages, sources, list(dict.fromkeys(warnings))


class DeadlineClient(httpx.Client):
    """Every model retry shares the same wall-clock budget, including revision cycles."""

    def __init__(self, deadline: float, per_call_timeout: float, **kwargs):
        super().__init__(**kwargs)
        self.deadline = deadline
        self.per_call_timeout = per_call_timeout

    def request(self, *args, **kwargs):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Interpretation deadline exceeded")
        kwargs["timeout"] = min(remaining, self.per_call_timeout)
        return super().request(*args, **kwargs)


def generate_interpretation(
    context: CompanyContext,
    news: NewsEvent,
    passages: list[BankPassage],
    settings: ServicesSettings,
    *,
    deadline: float | None = None,
    progress_callback: Callable[[str, dict], None] | None = None,
) -> InsightResult:
    require_llm(settings)
    progress_callback = progress_callback or _progress_callback.get()
    if settings.llm_mock:
        return build_insight(
            context,
            news,
            passages,
            llm=MockProvider(),
            settings=settings,
            progress_callback=progress_callback,
        )
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    with DeadlineClient(
        deadline=deadline or time.monotonic() + settings.interpretation_timeout_seconds,
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
        return build_insight(
            context,
            news,
            passages,
            llm=provider,
            settings=settings,
            progress_callback=progress_callback,
        )


def validate_result(
    result: InsightResult,
    context: CompanyContext,
    news: NewsEvent,
    sources: list[InterpretationSource],
) -> InsightResult:
    """Убирает непроверяемое из черновика вместо отказа от всего разбора.

    Неподтверждённая позиция банка и выдуманные ссылки удаляются, а в limitations
    остаётся пометка. Разбор признаётся недостаточным, только если после очистки не
    осталось ни одного утверждения с корректным источником.
    """

    result = result.model_copy(deep=True)
    draft = result.draft
    draft.company_id = context.company_id
    draft.event_id = news.url
    allowed = {s.evidence_ref for s in sources}
    allowed.update(f"company:{label}" for label in context.evidence_labels)
    notes: list[str] = []
    claims = []
    for claim in draft.claims:
        refs = [ref for ref in claim.evidence_refs if ref in allowed]
        if len(refs) != len(claim.evidence_refs):
            notes.append(f"У утверждения {claim.id} убраны ссылки на несуществующие источники.")
        if not refs:
            notes.append(f"Утверждение {claim.id} убрано: нет корректных ссылок на источники.")
            continue
        kind = claim.kind
        if kind == "bank_position" and not any(ref.startswith("bank:") for ref in refs):
            kind = "ai_inference"
            notes.append(f"Позиция банка в {claim.id} не подтверждена банковским источником.")
        claims.append(claim.model_copy(update={"evidence_refs": refs, "kind": kind}))
    draft.claims = claims
    if draft.bank_position and not any(
        claim.kind == "bank_position"
        and any(ref.startswith("bank:") for ref in claim.evidence_refs)
        for claim in claims
    ):
        draft.bank_position = None
        notes.append("Позиция банка не сформирована: нет подтверждённого материала банка.")
    if notes:
        draft.limitations = list(dict.fromkeys([*draft.limitations, *notes]))
    if not claims:
        result.verification.status = "INSUFFICIENT"
        result.verification.missing_evidence.append(
            "Для интерпретации не предоставлены утверждения с источниками."
        )
    return result


def completed_response(
    record_id,
    company_id,
    news_id,
    result: InsightResult,
    sources: list[InterpretationSource],
    warnings: list[str],
    settings: ServicesSettings,
) -> InterpretationResponse:
    status = {"APPROVED": "ready", "REVISE": "needs_revision", "INSUFFICIENT": "insufficient"}
    return InterpretationResponse(
        id=record_id,
        company_id=company_id,
        news_id=news_id,
        status=status[result.verification.status],
        interpretation=result.draft if result.verification.status == "APPROVED" else None,
        verification=result.verification,
        steps=result.steps,
        sources=sources,
        warnings=warnings
        + (
            ["Демонстрационный режим: ответ не является аналитикой LLM."]
            if settings.llm_mock
            else []
        ),
        model=settings.llm_model,
        is_demo=settings.llm_mock,
        generated_at=datetime.now(UTC),
    )
