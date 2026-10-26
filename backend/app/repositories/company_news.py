import asyncio
import re
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import Text, case, cast, func, literal, literal_column, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.company import Company
from app.models.news import NewsArticle
from app.models.news_embeddings import EMBEDDING_DIM, NewsEmbedding
from app.repositories.company_profiles import (
    load_current_company_profile,
    load_person_geo,
    prepare_company_profile_input,
)
from app.repositories.news_embeddings import (
    llm_model_version,
    load_ready_news_embeddings,
    news_input_fingerprint,
)
from app.schemas.company_news import CompanyNewsItem, CompanyNewsPage
from app.services.llm.config import ServicesSettings
from app.services.llm.matching import place_stems, score_relevance
from app.services.llm.news_analysis import NEWS_PROMPT_VERSION
from app.services.llm.rerank import get_reranker
from app.services.llm.schemas import NewsAnalysis, PersonGeo, ProcessedNews, RelevanceScore
from app.services.llm.text import shorten_text

BUSINESS_TOPICS = ("economy", "business", "finance", "taxes", "regulation")
# Процентная повестка, а не «ставки на перевозку нефти» или «ставка налога на игорный бизнес».
RATE_NEWS_PATTERN = (
    r"ключев\w* ставк|ставк\w* (цб|центробанка|по кредит|по вклад|по депозит|по ипотек)"
    r"|процентн\w* ставк"
)
GENERAL_RULE = "Общая деловая повестка; персональное влияние ещё не оценено"
STOPWORDS = {
    # слова из названий ОКВЭД, которые есть почти в любой отрасли и дают шум в поиске:
    # «производство» совпадало с новостями про вино, оборонку и сталь для любой компании
    "производство",
    "производства",
    "производств",
    "торговля",
    "торговли",
    "оптовая",
    "оптовой",
    "розничная",
    "розничной",
    "услуги",
    "услуг",
    "предоставление",
    "предоставления",
    "работы",
    "работ",
    "организаций",
    "организации",
    "изделий",
    "товаров",
    "средств",
    "средства",
    "иных",
    "других",
    "видов",
    "вида",
    "деятельность",
    "деятельности",
    "прочая",
    "прочие",
    "прочих",
    "прочий",
    "кроме",
    "включая",
}


def okved_keywords(okved_name: str | None) -> list[str]:
    """Значимые слова названия ОКВЭД для текстового правила (без общих слов)."""

    words = set(re.findall(r"[а-яёa-z]{4,}", (okved_name or "").lower())) - STOPWORDS
    return sorted(words)[:12]


def relevance_rules(company: Company, person_places: Sequence[PersonGeo] = ()):
    text = func.concat_ws(
        " ", NewsArticle.title, NewsArticle.summary, cast(NewsArticle.source_categories, Text)
    )
    rules = []
    words = okved_keywords(company.okved_name)
    if words:
        language = literal_column("'russian'")
        vector = func.to_tsvector(language, text)
        hits = sum(
            (
                case((vector.op("@@")(func.plainto_tsquery(language, word)), 1), else_=0)
                for word in words
            ),
            literal(0),
        )
        # Одно общее слово («аренда», «средств») совпадает с новостями любой тематики:
        # при трёх и более ключевых словах нужны хотя бы два совпадения.
        rules.append(
            (
                "Совпадение с описанием деятельности компании",
                hits >= (1 if len(words) <= 2 else 2),
                0.5,
            )
        )
    own_stems = place_stems(company.region) | place_stems(company.city)
    person_stems = sorted(
        {
            stem
            for person in person_places
            for place in (person.region, person.city)
            for stem in place_stems(place)
        }
        - own_stems
    )[:6]
    if person_stems:
        # Основы слов, а не названия целиком: «Тюмень» должна находить «в Тюмени».
        mentioned = or_(*(text.icontains(stem, autoescape=True) for stem in person_stems))
        rules.append(("Упоминание региона связанных физлиц", mentioned, 0.15))
    if company.region:
        rules.append(
            ("Упоминание региона компании", text.icontains(company.region, autoescape=True), 0.25)
        )
    if company.city:
        rules.append(
            ("Упоминание города компании", text.icontains(company.city, autoescape=True), 0.2)
        )
    if company.has_credit or company.has_deposit or company.has_leasing:
        condition = or_(
            NewsArticle.topics.contains(["finance"]), text.op("~*")(RATE_NEWS_PATTERN)
        )
        rules.append(("Финансовая повестка при наличии банковских продуктов", condition, 0.2))
    business = or_(*(NewsArticle.topics.contains([topic]) for topic in BUSINESS_TOPICS))
    rules.append((GENERAL_RULE, business, 0.1))
    return rules


async def list_company_news(
    session: AsyncSession,
    company: Company,
    *,
    limit: int,
    offset: int,
    source: str | None,
    topic: str | None,
    published_from: datetime | None,
    published_to: datetime | None,
    relevant_only: bool,
    settings: ServicesSettings,
) -> CompanyNewsPage:
    persons = await load_person_geo(session, company.id)
    rules = relevance_rules(company, persons[1])
    rule_score = sum(
        (case((condition, weight), else_=0.0) for _, condition, weight in rules), literal(0.0)
    )
    # Общая деловая повестка (0.1) лишь слегка поднимает новость в выдаче, но сама
    # не делает её «релевантной»: иначе в ленту попадает любая деловая новость.
    specific_score = sum(
        (
            case((condition, weight), else_=0.0)
            for name, condition, weight in rules
            if name != GENERAL_RULE
        ),
        literal(0.0),
    )
    filters = []
    if source:
        filters.append(NewsArticle.source_code == source)
    if topic:
        filters.append(NewsArticle.topics.contains([topic]))
    if published_from:
        filters.append(NewsArticle.published_at >= published_from)
    if published_to:
        filters.append(NewsArticle.published_at <= published_to)

    prepared = await prepare_company_profile_input(session, company, settings, persons)
    profile = await load_current_company_profile(session, prepared)
    semantic: dict[UUID, RelevanceScore] = {}
    semantic_articles: dict[UUID, tuple[NewsArticle, float, list[bool]]] = {}
    semantic_news: dict[UUID, ProcessedNews] = {}
    if profile is not None and len(profile.vectors.get("main", [])) == EMBEDDING_DIM:
        ready = _ready_embedding_conditions(profile.embedding_model_version, settings)
        candidate_rows = await session.execute(
            select(
                NewsEmbedding,
                NewsArticle,
                rule_score,
                *(condition for _, condition, _ in rules),
            )
            .join(NewsArticle, NewsArticle.id == NewsEmbedding.article_id)
            .where(*filters, *ready)
            .order_by(
                NewsEmbedding.vector.cosine_distance(profile.vectors["main"]),
                NewsArticle.first_seen_at.desc(),
                NewsArticle.id.desc(),
            )
            .limit(500)
        )
        candidate_values = candidate_rows.all()
        current_candidates = await load_ready_news_embeddings(
            session, [row[1] for row in candidate_values], settings
        )
        for embedding, article, rank, *matched in candidate_values:
            embedding = current_candidates.get(article.id)
            if embedding is None:
                continue
            processed = _processed_news(embedding, article)
            scored = score_relevance(
                profile,
                prepared.fields,
                processed,
                settings=settings,
                news_id=str(article.id),
            )
            semantic[article.id] = scored
            semantic_news[article.id] = processed
            semantic_articles[article.id] = (article, float(rank), list(matched))
        await _apply_reranker(profile, prepared.fields, semantic, semantic_news, settings)

    rule_filter = specific_score > 0
    if relevant_only:
        rule_total = int(
            await session.scalar(
                select(func.count()).select_from(NewsArticle).where(*filters, rule_filter)
            )
            or 0
        )
        semantic_only = sum(
            1
            for article_id, scored in semantic.items()
            if scored.is_relevant
            and not _has_specific_match(rules, semantic_articles[article_id][2])
        )
        total = rule_total + semantic_only
    else:
        total = int(
            await session.scalar(select(func.count()).select_from(NewsArticle).where(*filters)) or 0
        )

    # The SQL rules remain complete. Semantic scoring is bounded after all user filters are
    # applied, and its candidates are merged before the requested page is sliced.
    rule_rows = await session.execute(
        select(NewsArticle, rule_score, *(condition for _, condition, _ in rules))
        .where(*filters, *((rule_filter,) if relevant_only else ()))
        .order_by(rule_score.desc(), NewsArticle.first_seen_at.desc(), NewsArticle.id.desc())
        .limit(offset + limit + len(semantic))
    )
    combined: dict[UUID, tuple[NewsArticle, float, list[str]]] = {}
    for article, rank, *matched in rule_rows:
        reasons = [name for (name, _, _), yes in zip(rules, matched, strict=True) if yes]
        combined[article.id] = (article, float(rank), reasons)
    for article_id, scored in semantic.items():
        if relevant_only and not scored.is_relevant and article_id not in combined:
            continue
        article, rank, matched = semantic_articles[article_id]
        rule_reasons = [name for (name, _, _), yes in zip(rules, matched, strict=True) if yes]
        combined[article_id] = (
            article,
            max(rank, scored.score),
            list(dict.fromkeys([*scored.reasons, *rule_reasons])),
        )

    ranked = sorted(
        combined.values(),
        key=lambda item: (item[1], item[0].first_seen_at, str(item[0].id)),
        reverse=True,
    )[offset : offset + limit]
    articles = [article for article, _, _ in ranked]
    embeddings = await load_ready_news_embeddings(session, articles, settings)
    states = await _load_news_states(session, articles)
    assessments, pending_assessments = await _load_assessments(
        session, company, articles, settings
    )

    items = []
    for article, rank, reasons in ranked:
        embedding = embeddings.get(article.id)
        summary = embedding.summary if embedding and embedding.summary else article.summary
        assessment = assessments.get(article.id)
        items.append(
            CompanyNewsItem(
                id=article.id,
                source_code=article.source_code,
                title=article.title,
                summary=shorten_text(summary or article.title, sentences=2, chars=260),
                summary_origin="ai" if embedding and embedding.summary else (
                    "rss" if article.summary else "title"
                ),
                processing_status=_processing_status(embedding, states.get(article.id), settings),
                canonical_url=article.canonical_url,
                published_at=article.published_at,
                first_seen_at=article.first_seen_at,
                topics=article.topics,
                relevance_score=round(min(float(rank), 1.0), 3),
                relevance_reasons=reasons,
                assessment_status=(
                    "ready"
                    if assessment
                    else ("pending" if article.id in pending_assessments else "not_assessed")
                ),
                brief_impact=assessment.get("brief_impact") if assessment else None,
                risks=assessment.get("risks") if assessment else None,
                opportunities=assessment.get("opportunities") if assessment else None,
                bank_analytics=assessment.get("bank_analytics") if assessment else None,
                bank_sources=assessment.get("bank_sources", []) if assessment else [],
            )
        )
    return CompanyNewsPage(
        company_id=company.id,
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        ranking_method="hybrid" if profile is not None else "rules",
        personalization_status=(
            "ready"
            if profile is not None
            else ("preparing" if settings.profile_enabled else "unavailable")
        ),
        ranking_note=(
            "Семантическое ранжирование применяется максимум к 500 новостям после фильтров; "
            "совпадения по явным правилам учитываются без этого ограничения."
            if profile is not None
            else None
        ),
    )


def _has_specific_match(rules, matched: list[bool]) -> bool:
    return any(
        flag for (name, _, _), flag in zip(rules, matched, strict=True) if name != GENERAL_RULE
    )


async def _apply_reranker(
    profile,
    fields,
    semantic: dict[UUID, RelevanceScore],
    news: dict[UUID, ProcessedNews],
    settings: ServicesSettings,
) -> None:
    """Реранкер шлюза уточняет семантическую часть оценки у лучших кандидатов.

    При недоступности (или без AI_RERANK_MODEL) оценки остаются «косинус + веса правил».
    """
    reranker = get_reranker(settings)
    if not reranker.enabled or not semantic:
        return
    top = sorted(semantic, key=lambda item: semantic[item].score, reverse=True)[
        : settings.rerank_top_k
    ]
    scores = await asyncio.to_thread(
        reranker.rerank, profile.text.main_text, [news[item].summary for item in top]
    )
    for article_id, rerank in zip(top, scores, strict=True):
        if rerank is not None:
            semantic[article_id] = score_relevance(
                profile,
                fields,
                news[article_id],
                settings=settings,
                news_id=str(article_id),
                rerank=rerank,
            )


def _ready_embedding_conditions(model_version: str, settings: ServicesSettings) -> tuple:
    return (
        NewsEmbedding.status == "ready",
        NewsEmbedding.source_content_hash == NewsArticle.content_hash,
        NewsEmbedding.embedding_model_version == model_version,
        NewsEmbedding.llm_model_version == llm_model_version(settings),
        NewsEmbedding.prompt_version == NEWS_PROMPT_VERSION,
        NewsEmbedding.summary.is_not(None),
        NewsEmbedding.vector.is_not(None),
    )


def _processed_news(embedding: NewsEmbedding, article: NewsArticle) -> ProcessedNews:
    summary = embedding.summary or article.summary or article.title
    analysis = NewsAnalysis(**(embedding.analysis or {}), summary=summary)
    return ProcessedNews(
        url=article.original_url,
        text="\n\n".join(filter(None, [article.title, article.content_text or article.summary])),
        summary=summary,
        markup=analysis,
        vector=list(embedding.vector or []),
        embedding_model_version=embedding.embedding_model_version,
    )


def _processing_status(
    embedding: NewsEmbedding | None,
    state: NewsEmbedding | None,
    settings: ServicesSettings,
) -> str:
    if embedding is not None:
        return "ready"
    if state is not None and state.status == "processing":
        return "processing"
    return "pending" if settings.news_worker_enabled else "unavailable"


async def _load_news_states(
    session: AsyncSession, articles: list[NewsArticle]
) -> dict[UUID, NewsEmbedding]:
    if not articles:
        return {}
    rows = await session.scalars(
        select(NewsEmbedding).where(NewsEmbedding.article_id.in_([item.id for item in articles]))
    )
    return {row.article_id: row for row in rows}


async def _load_assessments(
    session: AsyncSession,
    company: Company,
    articles: list[NewsArticle],
    settings: ServicesSettings,
) -> tuple[dict, set[UUID]]:
    """Ready assessments by news id and ids whose short assessment is being prepared.

    Short assessments (risks, opportunities, bank analytics) are read from the DB; missing
    ones are queued for the background worker. A verified full interpretation fills the gap
    until the short one is ready.
    """
    if not articles:
        return {}, set()
    from app.repositories.assessments import enqueue_assessments, load_assessment_states
    from app.repositories.interpretations import load_ready_assessments
    from app.services.assessments import assessment_fingerprint, assessment_view, llm_is_configured
    from app.services.company_news import (
        company_input_fingerprint,
        load_company_context,
        model_prompt_version,
    )

    context = await load_company_context(session, company)
    company_fp = company_input_fingerprint(context)
    news_fingerprints = {
        article.id: news_input_fingerprint(article, settings) for article in articles
    }
    assessments: dict = await load_ready_assessments(
        session,
        company_id=company.id,
        news_fingerprints=news_fingerprints,
        company_input_fingerprint=company_fp,
        model_prompt_version=model_prompt_version(settings),
        trusted_bank_document_ids=settings.bank_document_ids,
    )
    pending: set[UUID] = set()
    if not settings.assessment_enabled:
        return assessments, pending
    fingerprints = {
        article.id: assessment_fingerprint(company_fp, article, settings) for article in articles
    }
    ready, queued = await load_assessment_states(
        session,
        company.id,
        fingerprints,
        trusted_bank_document_ids=settings.bank_document_ids,
    )
    for news_id, payload in ready.items():
        assessments[news_id] = assessment_view(payload)
    pending = queued - ready.keys()
    if llm_is_configured(settings):
        missing = {
            news_id: fingerprint
            for news_id, fingerprint in fingerprints.items()
            if news_id not in ready and news_id not in queued
        }
        limited = dict(list(missing.items())[: settings.assessment_enqueue_limit])
        pending |= await enqueue_assessments(session, company.id, limited)
    else:
        pending = set()
    return assessments, pending
