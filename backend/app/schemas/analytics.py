from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.analytics.parser import ParsedPage

DocumentStatus = Literal["queued", "extracting", "indexing", "parsed", "ready", "failed"]


class BankDocumentsSelection(BaseModel):
    document_ids: list[UUID] = Field(
        max_length=1000,
        description="Полный набор документов аналитики банка; [] отключает аналитику",
    )


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    sha256: str
    filename: str
    title: str
    publisher: str
    published_at: date | None
    date_origin: str
    topics: list[str]
    created_at: datetime
    status: DocumentStatus
    page_count: int
    chunk_count: int
    parser_version: str | None
    warnings: list[str]
    error: str | None
    embedding_error: str | None
    embedding_model_version: str | None


class UploadResponse(BaseModel):
    document: DocumentResponse
    duplicate: bool
    status_url: str


class DocumentUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=500)
    publisher: str | None = Field(default=None, min_length=1, max_length=300)
    published_at: date | None = None
    topics: list[str] | None = Field(default=None, max_length=30)

    @model_validator(mode="after")
    def validate_fields(self):
        for name in self.model_fields_set - {"published_at"}:
            value = getattr(self, name)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"{name} не должен быть пустым или null")
        if self.topics is not None and any(not t.strip() or len(t) > 100 for t in self.topics):
            raise ValueError("Тема должна содержать от 1 до 100 символов")
        return self


class PassageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    chunk_id: str
    page_number: int
    section: str
    text: str
    bbox: list[float]


class PageResponse(BaseModel):
    document_id: UUID
    page: ParsedPage


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    # Текст для смыслового поиска: если задан, эмбеддится он, а не `query` (который может
    # содержать операторы OR и слова профиля компании и сбивать поиск от темы новости).
    semantic_query: str | None = Field(default=None, max_length=4000)
    mode: Literal["hybrid", "keyword", "semantic"] = "hybrid"
    limit: int = Field(default=8, ge=1, le=30)
    publisher: str | None = Field(default=None, max_length=300)
    topic: str | None = Field(default=None, max_length=100)
    document_ids: list[UUID] = Field(default_factory=list, max_length=100)
    published_from: date | None = None
    as_of: date | None = None

    @model_validator(mode="after")
    def validate_period(self):
        if not self.query.strip():
            raise ValueError("Поисковый запрос не должен быть пустым")
        if self.published_from and self.as_of and self.published_from > self.as_of:
            raise ValueError("Начало периода должно предшествовать as_of")
        return self


class SearchHit(PassageResponse):
    title: str
    publisher: str
    published_at: date | None
    score: float
    # Косинус запроса и фрагмента (bge-m3); None, если фрагмент найден только по тексту.
    semantic_score: float | None = None
    evidence_ref: str
    source_url: str


class SearchResponse(BaseModel):
    items: list[SearchHit]
    mode_used: Literal["hybrid", "keyword", "semantic"]
    warnings: list[str] = Field(default_factory=list)
