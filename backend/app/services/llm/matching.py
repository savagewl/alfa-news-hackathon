import re

from app.services.llm.company_profile import PRODUCT_LABELS
from app.services.llm.config import ServicesSettings, get_services_settings
from app.services.llm.embeddings import cosine
from app.services.llm.schemas import (
    CompanyFields,
    CompanyProfile,
    ProcessedNews,
    RelevanceRule,
    RelevanceScore,
)

_SECTOR_EXACT_WEIGHT = 0.4
_SECTOR_WEIGHT = 0.35
_SECTOR_FUZZY_WEIGHT = 0.15
_REGION_WEIGHT = 0.2
_PERSONS_WEIGHT = 0.1
_COUNTERPARTY_WEIGHT = 0.25
_PRODUCT_WEIGHT = 0.15
_RATE_PRODUCT_WEIGHT = 0.15
_SCALE_WEIGHT = 0.1
# Только «банковские» сигналы (слово «кредит» в тексте, ставка) без отрасли, региона и
# контрагентов — слабая связь: вклад в оценку ограничен.
_BANKING_ONLY_CAP = 0.2

_REGION_STOPWORDS = {"область", "край", "республика", "округ", "город", "автономная", "мск"}

# Паттерны применяются к тексту в нижнем регистре; \b не даёт ловить «ярко», «внёс вклад» и т.п.
_PRODUCT_NEWS_PATTERNS = {
    "loa": re.compile(r"\b(кредит|финансировани|за[её]м|процентн)"),
    "lea": re.compile(r"\bлизинг"),
    "dep": re.compile(r"\b(депозит|вкладчик|вклады\b|банковск\w* вклад)"),
    "gar": re.compile(r"\b(гаранти|поручитель)"),
    "zpp": re.compile(r"\bзарплат"),
    "rko": re.compile(r"\b(рко\b|расч[её]тн\w* сч[её]т|эквайринг)"),
}

_MIN_COUNTERPARTY_NAME = 3
_REGION_STEM_LEN = 5

_ANY_SIZE = "любой"

_SCALE_MATCHES = {
    "менее 15": {"малый", "микро", "любой"},
    "16-100": {"малый", "средний", "любой"},
    "101-250": {"средний", "крупный", "любой"},
    "251+": {"крупный", "любой"},
}


def score_relevance(
    profile: CompanyProfile,
    fields: CompanyFields,
    news: ProcessedNews,
    *,
    settings: ServicesSettings | None = None,
    news_id: str | None = None,
    rerank: float | None = None,
) -> RelevanceScore:
    settings = settings or get_services_settings()
    cos = cosine(profile.vectors.get("main", []), news.vector)
    rules: list[RelevanceRule] = []
    reasons: list[str] = []

    sector_rule = _sector_rule(fields, news)
    region_rule = _region_rule(fields, news)
    persons_rule = _persons_rule(fields, news, region_rule)
    counterparty_rule = _counterparty_rule(fields, news)
    product_rule = _product_rule(fields, news)
    rate_rule = _rate_product_rule(fields, news)
    scale_rule = _scale_rule(fields, news)
    rules.extend(
        [
            sector_rule,
            region_rule,
            persons_rule,
            counterparty_rule,
            product_rule,
            rate_rule,
            scale_rule,
        ]
    )

    for rule in rules:
        if rule.matched:
            reasons.append(f"{rule.detail} (+{rule.weight})")
    if rerank is None:
        reasons.append(f"Семантическое сходство: {cos:.2f}")
        semantic = max(cos, 0.0)
    else:
        # Оценка реранкера заменяет косинус; правила и веса остаются прежними.
        reasons.append(f"Семантическое сходство (реранкер): {rerank:.2f}")
        semantic = max(rerank, 0.0)

    matched_total = sum(rule.weight for rule in rules if rule.matched)
    entity_matched = any(
        rule.matched for rule in (sector_rule, region_rule, persons_rule, counterparty_rule)
    )
    banking_total = sum(rule.weight for rule in (product_rule, rate_rule) if rule.matched)
    if not entity_matched and banking_total > _BANKING_ONLY_CAP:
        matched_total -= banking_total - _BANKING_ONLY_CAP
    score = settings.relevance_cosine_weight * semantic + matched_total
    score = min(score, 1.0)
    return RelevanceScore(
        company_id=fields.synthetic_u,
        news_id=news_id,
        news_url=news.url,
        cosine=cos,
        score=score,
        is_relevant=score >= settings.relevance_threshold,
        rules=rules,
        reasons=reasons,
    )


def _sector_rule(fields: CompanyFields, news: ProcessedNews) -> RelevanceRule:
    """Совпадение отрасли по иерархии ОКВЭД: подкласс > группа > класс (смежные подотрасли).

    Класс «25» включает и металлоконструкции (25.11), и оружие (25.4), поэтому совпадение
    только по двум цифрам — слабый сигнал и сам по себе порог релевантности не проходит.
    """

    company_code = (fields.okved_code or "").strip()
    company_class, company_group = _sector_code(company_code), _sector_group(company_code)
    news_codes = [sector.code.strip() for sector in news.markup.affected_sectors if sector.code]
    if company_code and company_code in news_codes:
        return RelevanceRule(
            name="sector",
            matched=True,
            weight=_SECTOR_EXACT_WEIGHT,
            detail=f"Совпадение подкласса ОКВЭД {company_code}",
        )
    if company_group and company_group in {_sector_group(code) for code in news_codes}:
        return RelevanceRule(
            name="sector",
            matched=True,
            weight=_SECTOR_WEIGHT,
            detail=f"Совпадение группы ОКВЭД {company_group}",
        )
    # Новость размечена целым классом («25») без подкласса — она про отрасль в целом.
    if company_class and company_class in {code for code in news_codes if "." not in code}:
        return RelevanceRule(
            name="sector",
            matched=True,
            weight=_SECTOR_WEIGHT,
            detail=f"Совпадение класса ОКВЭД {company_class}",
        )
    if company_class and company_class in {_sector_code(code) for code in news_codes}:
        return RelevanceRule(
            name="sector",
            matched=True,
            weight=_SECTOR_FUZZY_WEIGHT,
            detail=f"Смежная отрасль в классе ОКВЭД {company_class}",
        )
    if news.markup.affected_sectors and fields.okved_name:
        company_words = _meaningful_words(fields.okved_name)
        for sector in news.markup.affected_sectors:
            if company_words & _meaningful_words(sector.name):
                return RelevanceRule(
                    name="sector",
                    matched=True,
                    weight=_SECTOR_FUZZY_WEIGHT,
                    detail=f"Отраслевое совпадение: {sector.name}",
                )
    return RelevanceRule(
        name="sector",
        matched=False,
        weight=_SECTOR_WEIGHT,
        detail="нет совпадения отрасли",
    )


def _region_rule(fields: CompanyFields, news: ProcessedNews) -> RelevanceRule:
    company_tokens = _region_tokens(fields.region) | _region_tokens(fields.city)
    if not company_tokens:
        return RelevanceRule(
            name="region", matched=False, weight=_REGION_WEIGHT, detail="регион неизвестен"
        )
    for region in news.markup.regions:
        news_tokens = _region_tokens(region)
        if company_tokens & news_tokens:
            return RelevanceRule(
                name="region",
                matched=True,
                weight=_REGION_WEIGHT,
                detail=f"Региональная связь: {region}",
            )
    return RelevanceRule(
        name="region",
        matched=False,
        weight=_REGION_WEIGHT,
        detail="нет региональной связи",
    )


def _persons_rule(
    fields: CompanyFields, news: ProcessedNews, region_rule: RelevanceRule
) -> RelevanceRule:
    """Регион/город связанных физлиц: новость о нём важна компании, даже если её регион другой."""

    if region_rule.matched:
        return RelevanceRule(
            name="persons",
            matched=False,
            weight=_PERSONS_WEIGHT,
            detail="связь по региону компании уже учтена",
        )
    news_tokens = set()
    for region in news.markup.regions:
        news_tokens |= _region_tokens(region)
    haystack = (news.text + " " + news.summary).lower()
    for person in fields.persons:
        place = _region_tokens(person.region) | _region_tokens(person.city)
        text_stems = place_stems(person.region) | place_stems(person.city)
        if place & news_tokens or any(stem in haystack for stem in text_stems):
            label = ", ".join(part for part in (person.city, person.region) if part)
            return RelevanceRule(
                name="persons",
                matched=True,
                weight=_PERSONS_WEIGHT,
                detail=f"Региональная связь через связанных физлиц: {label}",
            )
    return RelevanceRule(
        name="persons",
        matched=False,
        weight=_PERSONS_WEIGHT,
        detail="регионы связанных физлиц не затронуты",
    )


def _counterparty_rule(fields: CompanyFields, news: ProcessedNews) -> RelevanceRule:
    haystack = f"{news.text}\n{news.summary}".lower()
    for counterparty in fields.counterparties:
        name = _counterparty_key(counterparty.name)
        if len(name) < _MIN_COUNTERPARTY_NAME:
            continue
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", haystack):
            return RelevanceRule(
                name="counterparty",
                matched=True,
                weight=_COUNTERPARTY_WEIGHT,
                detail=f"Упомянут контрагент: {counterparty.name}",
            )
    return RelevanceRule(
        name="counterparty",
        matched=False,
        weight=_COUNTERPARTY_WEIGHT,
        detail="контрагенты не упомянуты",
    )


def _product_rule(fields: CompanyFields, news: ProcessedNews) -> RelevanceRule:
    haystack = f"{news.text}\n{news.summary}".lower()
    for product in fields.products:
        pattern = _PRODUCT_NEWS_PATTERNS.get(product)
        if pattern and pattern.search(haystack):
            return RelevanceRule(
                name="product",
                matched=True,
                weight=_PRODUCT_WEIGHT,
                detail=f"Новость затрагивает продукт: {_product_label(product)}",
            )
    return RelevanceRule(
        name="product",
        matched=False,
        weight=_PRODUCT_WEIGHT,
        detail="нет продуктовой связи",
    )


def _rate_product_rule(fields: CompanyFields, news: ProcessedNews) -> RelevanceRule:
    money_products = {"loa", "dep", "lea"}
    if "rate" in news.markup.impact_kinds and any(
        product in money_products for product in fields.products
    ):
        return RelevanceRule(
            name="rate_products",
            matched=True,
            weight=_RATE_PRODUCT_WEIGHT,
            detail="Динамика ключевой ставки влияет на кредитные/депозитные продукты компании",
        )
    return RelevanceRule(
        name="rate_products",
        matched=False,
        weight=_RATE_PRODUCT_WEIGHT,
        detail="нет связи со ставкой",
    )


def _scale_rule(fields: CompanyFields, news: ProcessedNews) -> RelevanceRule:
    company_scale = _SCALE_MATCHES.get((fields.workers_fns or "").strip().lower())
    if not company_scale or not news.markup.size_relevance:
        return RelevanceRule(
            name="scale",
            matched=False,
            weight=_SCALE_WEIGHT,
            detail="масштаб не сопоставлен",
        )
    for size in news.markup.size_relevance:
        # «любой» — значение по умолчанию (в новости нет оснований): масштаб не сопоставляем.
        if size.strip().lower() != _ANY_SIZE and size.strip().lower() in company_scale:
            return RelevanceRule(
                name="scale",
                matched=True,
                weight=_SCALE_WEIGHT,
                detail=f"Масштаб релевантен: {size}",
            )
    return RelevanceRule(
        name="scale", matched=False, weight=_SCALE_WEIGHT, detail="масштаб не совпал"
    )


def _sector_group(code: str | None) -> str:
    """Группа ОКВЭД: «25.11» -> «25.1»; для кода без подкласса пустая строка."""

    value = (code or "").strip()
    return value[:4] if "." in value and len(value) >= 4 else ""


def _sector_code(code: str | None) -> str:
    return (code or "").split(".")[0].strip()


_SECTOR_STOPWORDS = {"деятельность", "производство", "прочая", "связанная"}


def _meaningful_words(value: str) -> set[str]:
    words = re.findall(r"[а-яё]{5,}", value.lower())
    return {word[:5] for word in words if word not in _SECTOR_STOPWORDS}


def _region_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {
        _region_stem(word)
        for word in re.findall(r"[а-яё]{4,}", value.lower())
        if word not in _REGION_STOPWORDS
    }


def place_stems(value: str | None) -> set[str]:
    """Основы для поиска в тексте: «Тюмень», «Тюмени», «Тюменская» дают «тюмен»."""

    if not value:
        return set()
    return {
        word[:_REGION_STEM_LEN]
        for word in re.findall(r"[а-яё]{4,}", value.lower())
        if word not in _REGION_STOPWORDS
    }


def _region_stem(word: str) -> str:
    """Грубая основа слова: «Курган»/«Курганская», «Москва»/«Московская» дают один токен."""
    if word.startswith(("москв", "моско")):
        return "москва"
    return word[:_REGION_STEM_LEN]


def _counterparty_key(name: str | None) -> str:
    """Имя контрагента без «ООО»/кавычек для поиска в тексте новости."""
    cleaned = re.sub(r"[«»\"'()]", " ", (name or "").lower())
    cleaned = re.sub(r"\b(ооо|оао|зао|пао|ао|ип)\b", " ", cleaned)
    return " ".join(cleaned.split())


def _product_label(product: str) -> str:
    return PRODUCT_LABELS.get(product, product)
