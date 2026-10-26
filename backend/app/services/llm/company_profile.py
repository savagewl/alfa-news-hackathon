from app.services.llm.config import ServicesSettings, get_services_settings
from app.services.llm.embeddings import EmbeddingsClient, get_embedder
from app.services.llm.schemas import CompanyFields, CompanyProfile, CompanyProfileText, Counterparty

PRODUCT_LABELS = {
    "rko": "расчётно-кассовое обслуживание",
    "gar": "банковские гарантии",
    "loa": "кредитование",
    "lea": "лизинг",
    "dep": "депозиты",
    "zpp": "зарплатный проект",
}

_UNKNOWN = "неизвестно"


def _group(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def format_revenue(value: float | None) -> str:
    if value is None:
        return _UNKNOWN
    if value < 1_000_000:
        return f"{_group(value)} руб."
    if value < 1_000_000_000:
        return f"{value / 1_000_000:.1f} млн руб."
    return f"{value / 1_000_000_000:.1f} млрд руб."


def build_company_profile_text(
    fields: CompanyFields,
    counterparties: list[Counterparty] | None = None,
) -> CompanyProfileText:
    parts = [
        f"Компания {fields.synthetic_u}",
        f"Тип организации: {_v(fields.org_type)}",
        f"Форма собственности: {_v(fields.okfs_name)}",
        f"Возраст: {_years(fields.company_age_years)}",
        f"Стадия развития: {_v(fields.eco_stage)}",
        _industry_part(fields),
        f"Численность сотрудников: {_v(fields.workers_fns)}",
        f"Тип клиента: {_v(fields.client_type)}",
        f"Продукты банка: {_products(fields)}",
        f"Выручка: {format_revenue(fields.revenue)}",
        f"Негативная метка: {_negative(fields.negative_mark)}",
        f"Факторинг: {_used(fields.factoring)}",
        f"ВЭД: {_fts(fields)}",
        f"География: {_geo(fields)}",
    ]
    if fields.group_id:
        parts.append(f"Группа компаний: {fields.group_id}")
    main_text = "; ".join(parts)
    if counterparties is None:
        counterparties = fields.counterparties
    if counterparties:
        main_text += "; Контрагенты: " + _counterparties_block(counterparties)
    aspects = _aspects(fields)
    return CompanyProfileText(synthetic_u=fields.synthetic_u, main_text=main_text, aspects=aspects)


def profile_field_labels(
    fields: CompanyFields, counterparties: list[Counterparty] | None = None
) -> list[str]:
    """Метки полей, как они напечатаны в build_company_profile_text (для evidence_refs)."""
    labels = [
        "Тип организации",
        "Форма собственности",
        "Возраст",
        "Стадия развития",
        "Численность сотрудников",
        "Тип клиента",
        "Продукты банка",
        "Выручка",
        "Негативная метка",
        "Факторинг",
        "ВЭД",
        "География",
    ]
    if fields.okved_code and fields.okved_name:
        labels.append(f"Отрасль (ОКВЭД {fields.okved_code})")
    else:
        labels.append("Отрасль")
    if fields.group_id:
        labels.append("Группа компаний")
    if counterparties if counterparties is not None else fields.counterparties:
        labels.append("Контрагенты")
    return labels


def embed_company_profile(
    text: CompanyProfileText,
    *,
    embedder: EmbeddingsClient | None = None,
    settings: ServicesSettings | None = None,
) -> dict[str, list[float]]:
    settings = settings or get_services_settings()
    embedder = embedder or get_embedder()
    texts = [text.main_text, *text.aspects.values()]
    vectors = embedder.embed(texts)
    keys = ["main", *text.aspects.keys()]
    return dict(zip(keys, vectors, strict=False))


def build_company_profile(
    fields: CompanyFields,
    counterparties: list[Counterparty] | None = None,
    *,
    embedder: EmbeddingsClient | None = None,
) -> CompanyProfile:
    text = build_company_profile_text(fields, counterparties)
    vectors = embed_company_profile(text, embedder=embedder)
    return CompanyProfile(
        text=text,
        vectors=vectors,
        embedding_model_version=(embedder or get_embedder()).model_version,
    )


def _v(value: str | None) -> str:
    return value if value else _UNKNOWN


def _years(value: str | None) -> str:
    return f"{value} лет" if value else _UNKNOWN


def _industry_part(fields: CompanyFields) -> str:
    if fields.okved_code and fields.okved_name:
        return f"Отрасль (ОКВЭД {fields.okved_code}): {fields.okved_name}"
    if fields.okved_name:
        return f"Отрасль: {fields.okved_name}"
    return f"Отрасль: {_UNKNOWN}"


def _products(fields: CompanyFields) -> str:
    if not fields.products:
        return _UNKNOWN
    return ", ".join(PRODUCT_LABELS.get(code, code) for code in fields.products)


def _used(flag: bool) -> str:
    return "используется" if flag else "нет"


def _negative(value: str | None) -> str:
    stripped = (value or "").strip()
    if not stripped or stripped in {"0", "0.0"}:
        return "нет"
    return "есть (причина неизвестна)"


def _fts(fields: CompanyFields) -> str:
    if not fields.fts_flag:
        return "нет"
    country = fields.fts_country_name or "неизвестно"
    return f"да, страны: {country}"


def _geo(fields: CompanyFields) -> str:
    pieces = [piece for piece in (fields.city, fields.region) if piece]
    return ", ".join(pieces) if pieces else _UNKNOWN


def _counterparties_block(counterparties: list[Counterparty]) -> str:
    top = sorted(counterparties, key=lambda c: c.pay_amt, reverse=True)[:5]
    items = []
    for cp in top:
        direction = "входящие" if cp.direction == "incoming" else "исходящие"
        items.append(f"{cp.name} ({direction}, {format_revenue(cp.pay_amt)}, {cp.pay_cnt} плат.)")
    return ", ".join(items)


def _aspects(fields: CompanyFields) -> dict[str, str]:
    industry = (
        f"Отрасль: ОКВЭД {fields.okved_code} — {fields.okved_name}"
        if fields.okved_code and fields.okved_name
        else f"Отрасль: {_v(fields.okved_name)}"
    )
    industry += (
        f"; численность: {_v(fields.workers_fns)}; стадия: {_v(fields.eco_stage)}; "
        f"география: {_geo(fields)}"
    )
    finance = (
        f"Выручка: {format_revenue(fields.revenue)}; тип клиента: {_v(fields.client_type)}; "
        f"продукты банка: {_products(fields)}; факторинг: {_used(fields.factoring)}"
    )
    ved = f"Экспортная деятельность: {_fts(fields)}; отрасль: {_v(fields.okved_name)}"
    return {"industry": industry, "finance": finance, "ved": ved}
