from app.services.llm.config import ServicesSettings, get_services_settings
from app.services.llm.embeddings import EmbeddingsClient, get_embedder
from app.services.llm.provider import LLMProvider, complete_json, get_llm_provider, messages
from app.services.llm.schemas import ImpactKind, NewsAnalysis, OkvedSector, ProcessedNews

NEWS_SYSTEM_PROMPT = (
    "Ты — аналитик деловых новостей Альфа-Банка. Проанализируй новость и верни строго JSON, "
    "без markdown и пояснений:\n"
    "{\n"
    '  "summary": "2-4 предложения: что произошло, кто участники, цифры, сроки",\n'
    '  "topics": ["тема1", "тема2"],\n'
    '  "affected_sectors": [{"code": "код ОКВЭД или null", '
    '"name": "название отрасли по-русски"}],\n'
    '  "regions": ["регионы в именительном падеже, например Россия"],\n'
    '  "impact_kinds": ["rate", "currency", "taxes", "logistics", "industry", "regulation", '
    '"demand", "energy", "trade_restrictions", "other"],\n'
    '  "facts": ["короткие фактические утверждения из текста новости"],\n'
    '  "size_relevance": ["крупный", "средний", "малый", "любой"]\n'
    "}\n"
    "Правила: используй только информацию из текста новости, без домыслов; "
    "facts — дословные суть-факты; affected_sectors — только реально затронутые отрасли; "
    "impact_kinds — только из разрешённого списка; "
    "size_relevance отмечай только если в тексте есть основания "
    "(численность сотрудников, масштаб компаний, например крупный/малый); "
    "если в новости нет конкретики (суммы, даты, имена, нормативные акты) — "
    "не усиливай значимость домыслами."
)

# Bump whenever NEWS_SYSTEM_PROMPT or the output semantics change.  It is part
# of the durable news fingerprint, so old summaries are never silently reused.
NEWS_PROMPT_VERSION = "news-analysis-v1"

_KNOWN_IMPACTS = {kind.value.lower(): kind.value for kind in ImpactKind}
_SIZE_ALIASES = {
    "large": "крупный",
    "big": "крупный",
    "medium": "средний",
    "middle": "средний",
    "small": "малый",
    "micro": "малый",
    "any": "любой",
    "all": "любой",
}


def analyze_news_text(
    text: str,
    *,
    llm: LLMProvider | None = None,
    settings: ServicesSettings | None = None,
) -> NewsAnalysis:
    settings = settings or get_services_settings()
    llm = llm or get_llm_provider(settings)
    messages_list = messages(NEWS_SYSTEM_PROMPT, f"Новость:\n{text[: settings.news_max_chars]}")
    analysis, _ = complete_json(
        "news_analysis",
        NewsAnalysis,
        messages_list,
        provider=llm,
        settings=settings,
        temperature=settings.llm_news_temperature,
    )
    return _normalize(analysis)


def process_news_text(
    text: str,
    url: str | None = None,
    *,
    llm: LLMProvider | None = None,
    embedder: EmbeddingsClient | None = None,
    settings: ServicesSettings | None = None,
) -> ProcessedNews:
    settings = settings or get_services_settings()
    analysis = analyze_news_text(text, llm=llm, settings=settings)
    embedder = embedder or get_embedder()
    vector = embedder.embed_one(analysis.summary)
    return ProcessedNews(
        url=url,
        text=text,
        summary=analysis.summary,
        markup=analysis,
        vector=vector,
        embedding_model_version=embedder.model_version,
    )


def _normalize(analysis: NewsAnalysis) -> NewsAnalysis:
    impacts: list[str] = []
    for kind in analysis.impact_kinds:
        value = _KNOWN_IMPACTS.get(kind.strip().lower())
        if value and value not in impacts:
            impacts.append(value)
    if not impacts:
        impacts = [ImpactKind.OTHER.value]
    sizes: list[str] = []
    for size in analysis.size_relevance:
        normalized = _SIZE_ALIASES.get(size.strip().lower())
        if not normalized and size.strip():
            normalized = size.strip().lower()
        if normalized and normalized not in sizes:
            sizes.append(normalized)
    if not sizes:
        sizes = ["любой"]  # без оснований в тексте новость применима к любой компании
    return analysis.model_copy(
        update={
            "impact_kinds": impacts,
            "size_relevance": sizes,
            "affected_sectors": _clean_sectors(analysis.affected_sectors),
        }
    )


def _clean_sectors(sectors: list[OkvedSector]) -> list[OkvedSector]:
    cleaned = []
    for sector in sectors:
        name = sector.name.strip()
        code = sector.code.strip() if sector.code else ""
        if not name and not code:
            continue
        cleaned.append(OkvedSector(code=code or None, name=name or "неизвестная отрасль"))
    return cleaned
