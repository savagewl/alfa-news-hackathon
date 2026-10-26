from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def _as_model_input(value):
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item is not None]
    if isinstance(value, (dict, BaseModel)):
        return [value]
    return []


class OkvedSector(BaseModel):
    code: str | None = None
    name: str


class NewsAnalysis(BaseModel):
    summary: str = Field(min_length=1)
    topics: list[str] = Field(default_factory=list)
    affected_sectors: list[OkvedSector] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    impact_kinds: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    size_relevance: list[str] = Field(default_factory=list)

    _coerce_lists = field_validator(
        "topics", "regions", "impact_kinds", "facts", "size_relevance", mode="before"
    )(_as_model_input)
    _coerce_sectors = field_validator("affected_sectors", mode="before")(_as_model_input)


class ProcessedNews(BaseModel):
    url: str | None = None
    text: str
    summary: str
    markup: NewsAnalysis
    vector: list[float]
    embedding_model_version: str


class CompanyFields(BaseModel):
    synthetic_u: str
    org_type: str | None = None
    okfs_name: str | None = None
    company_age_years: str | None = None
    eco_stage: str | None = None
    okved_code: str | None = None
    okved_name: str | None = None
    negative_mark: str | None = None
    workers_fns: str | None = None
    workers_spark: str | None = None
    client_type: str | None = None
    products: list[str] = Field(default_factory=list)
    fts_flag: bool = False
    fts_country_name: str | None = None
    factoring: bool = False
    revenue: float | None = None
    group_id: str | None = None
    city: str | None = None
    region: str | None = None
    counterparties: list["Counterparty"] = Field(default_factory=list)

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> "CompanyFields":
        product_marks = ["rko", "gar", "loa", "lea", "dep", "zpp"]
        products = [
            mark
            for mark in product_marks
            if _to_float(row.get(f"CL_PRODUCT_{mark.upper()}_MARK")) == 1.0
        ]
        return cls(
            synthetic_u=row["SYNTHETIC_U"].strip(),
            org_type=_clean(row.get("ORG_TYPE")),
            okfs_name=_clean(row.get("OKFS_NAME")),
            company_age_years=_clean(row.get("COMPANY_AGE_YEARS_GRADATION")),
            eco_stage=_clean(row.get("ECO_MARK_LABEL")),
            okved_code=_clean(row.get("MAINOKVED_CODE")),
            okved_name=_clean(row.get("MAINOKVED_NAME")),
            negative_mark=_clean(row.get("NEGATIVEMARK")),
            workers_fns=_clean(row.get("AMOUNT_WORKERS_FNS_GRADATION")),
            workers_spark=_clean(row.get("AMOUNT_WORKERS_SPARK_GRADATION")),
            client_type=_clean(row.get("CL_TYPE")),
            products=products,
            fts_flag=_to_float(row.get("FTS_FLAG")) == 1.0,
            fts_country_name=_clean(row.get("FTS_COUNTRY_NAME")),
            factoring=_to_float(row.get("FACTORING_MARK")) == 1.0,
            revenue=_to_float(row.get("REVENUE_COMPANY")),
            group_id=_clean(row.get("GROUP_ID")),
            city=_clean(row.get("CITY")),
            region=_clean(row.get("REGION")),
        )


class Counterparty(BaseModel):
    name: str
    direction: Literal["incoming", "outgoing"] = "incoming"
    pay_amt: float = 0.0
    pay_cnt: int = 0


class CompanyProfileText(BaseModel):
    synthetic_u: str
    main_text: str
    aspects: dict[str, str]


class CompanyProfile(BaseModel):
    text: CompanyProfileText
    vectors: dict[str, list[float]]
    embedding_model_version: str


class RelevanceRule(BaseModel):
    name: str
    matched: bool
    weight: float
    detail: str


class RelevanceScore(BaseModel):
    company_id: str
    news_id: str | None = None
    news_url: str | None = None
    cosine: float
    score: float
    is_relevant: bool
    rules: list[RelevanceRule]
    reasons: list[str]


class AssessmentDraft(BaseModel):
    """Короткая оценка новости для компании: каждый блок — 1–2 коротких предложения.

    Если значимых рисков или возможностей нет, блок равен None — фронт его не показывает.
    """

    brief_impact: str | None = None
    risks: str | None = None
    opportunities: str | None = None
    bank_analytics: str | None = None
    # Номера фрагментов [B#] из блока «Аналитика Альфа-Банка», на которые опирается текст.
    bank_refs: list[int] = Field(default_factory=list)

    @field_validator("brief_impact", "risks", "opportunities", "bank_analytics", mode="after")
    @classmethod
    def _empty_to_none(cls, value: str | None) -> str | None:
        from app.services.llm.text import is_empty_assessment

        return None if is_empty_assessment(value) else value.strip()


class BankPassage(BaseModel):
    document_id: str
    chunk_id: str | None = None
    title: str
    text: str
    published_at: str | None = None


class NewsEvent(BaseModel):
    url: str | None = None
    title: str | None = None
    text: str
    summary: str
    markup: NewsAnalysis | None = None
    published_at: str | None = None


class CompanyContext(BaseModel):
    company_id: str
    profile_text: str
    okved_code: str | None = None
    okved_name: str | None = None
    region: str | None = None
    city: str | None = None
    client_type: str | None = None
    products: list[str] = Field(default_factory=list)
    counterparties_text: str | None = None
    relevance_reasons: list[str] = Field(default_factory=list)
    as_of: str | None = None
    evidence_labels: list[str] = Field(default_factory=list)


class InsightClaim(BaseModel):
    id: str = ""
    text: str = ""
    kind: Literal["fact", "bank_position", "ai_inference"]
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value) -> str:
        lowered = str(value).strip().lower()
        if lowered in {"ai", "inference", "ai_inference"}:
            return "ai_inference"
        if lowered in {"bank", "bank_position"}:
            return "bank_position"
        return lowered

    @field_validator("evidence_refs", mode="before")
    @classmethod
    def _coerce_evidence(cls, value) -> list:
        return _as_model_input(value)


class InsightDraft(BaseModel):
    event_id: str | None = None
    company_id: str | None = None
    headline: str = Field(min_length=1)
    facts: list[str] = Field(default_factory=list)
    bank_position: str | None = None
    ai_interpretation: str
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    suggested_actions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    claims: list[InsightClaim] = Field(default_factory=list)

    _coerce_lists = field_validator(
        "facts",
        "risks",
        "opportunities",
        "suggested_actions",
        "limitations",
        "claims",
        mode="before",
    )(_as_model_input)

    @field_validator("claims", mode="after")
    @classmethod
    def _fill_missing_claim_fields(cls, value: list[InsightClaim]) -> list[InsightClaim]:
        filled = []
        for index, claim in enumerate(value, start=1):
            filled.append(
                claim.model_copy(
                    update={
                        "id": claim.id or f"c{index}",
                        "text": claim.text or "не сформулировано",
                    }
                )
            )
        return filled

    @field_validator("headline", "ai_interpretation", mode="before")
    @classmethod
    def _coerce_text_fields(cls, value) -> str:
        if not isinstance(value, str) or not value.strip():
            return "Модель не сформулировала вывод в этом поле"
        return value


class Verification(BaseModel):
    status: Literal["APPROVED", "REVISE", "INSUFFICIENT"]
    remarks: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    retry_queries: list[str] = Field(default_factory=list)

    _coerce_lists = field_validator("remarks", "missing_evidence", "retry_queries", mode="before")(
        _as_model_input
    )

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, value) -> str:
        normalized = str(value).strip().upper()
        aliases = {"OK": "APPROVED", "REVISED": "REVISE", "RETRY": "REVISE", "REWRITE": "REVISE"}
        return aliases.get(normalized, normalized)


class InsightStep(BaseModel):
    step: str
    status: str
    tokens_prompt: int = 0
    tokens_completion: int = 0
    detail: str = ""


class InsightResult(BaseModel):
    draft: InsightDraft
    verification: Verification
    steps: list[InsightStep]


class ImpactKind(StrEnum):
    RATE = "rate"
    CURRENCY = "currency"
    TAXES = "taxes"
    LOGISTICS = "logistics"
    INDUSTRY = "industry"
    REGULATION = "regulation"
    DEMAND = "demand"
    ENERGY = "energy"
    TRADE_RESTRICTIONS = "trade_restrictions"
    OTHER = "other"


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _to_float(value: str | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
