from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.services.llm.schemas import InsightDraft, InsightStep, Verification


class InterpretationSource(BaseModel):
    evidence_ref: str
    title: str
    source_url: str
    published_at: str | None = None
    page_number: int | None = None


class CompanyNewsItem(BaseModel):
    id: UUID
    source_code: str
    title: str
    summary: str
    summary_origin: Literal["ai", "rss", "title"]
    processing_status: Literal["ready", "processing", "pending", "unavailable"] = "pending"
    canonical_url: str
    published_at: datetime | None
    first_seen_at: datetime
    topics: list[str]
    relevance_score: float
    relevance_reasons: list[str]
    # ready — оценка есть; pending — готовится фоном (повторите запрос ленты); not_assessed — нет.
    assessment_status: Literal["ready", "pending", "not_assessed"] = "not_assessed"
    brief_impact: str | None = None
    # Короткие блоки по 2-4 предложения (один элемент списка) или пункты полной интерпретации.
    risks: list[str] | None = None
    opportunities: list[str] | None = None
    # Позиция аналитиков банка только по найденным фрагментам PDF; null, если материалов нет.
    bank_analytics: str | None = None
    bank_sources: list[InterpretationSource] = []


class CompanyNewsPage(BaseModel):
    company_id: str
    items: list[CompanyNewsItem]
    total: int
    limit: int
    offset: int
    ranking_method: Literal["rules", "hybrid"] = "rules"
    personalization_status: Literal["ready", "preparing", "unavailable"] = "preparing"
    ranking_note: str | None = None


class InterpretationResponse(BaseModel):
    id: UUID
    company_id: str
    news_id: UUID
    status: Literal["generating", "ready", "needs_revision", "insufficient"]
    interpretation: InsightDraft | None = None
    verification: Verification | None = None
    steps: list[InsightStep] = []
    sources: list[InterpretationSource] = []
    warnings: list[str] = []
    model: str
    is_demo: bool = False
    reused: bool = False
    generated_at: datetime | None = None


class AssessmentStreamRequest(BaseModel):
    news_ids: list[UUID] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def unique_news_ids(self):
        if len(set(self.news_ids)) != len(self.news_ids):
            raise ValueError("news_ids must be unique")
        return self
