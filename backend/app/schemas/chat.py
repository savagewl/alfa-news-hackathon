from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatSource(BaseModel):
    evidence_ref: str
    kind: Literal["news", "bank", "company"]
    title: str
    source_url: str | None = None
    published_at: str | None = None
    page_number: int | None = None


class ChatThreadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    company_id: str
    news_id: UUID
    created_at: datetime
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    id: UUID
    thread_id: UUID
    client_message_id: UUID | None = None
    reply_to_id: UUID | None = None
    role: Literal["user", "assistant"]
    status: Literal["generating", "ready", "failed"] = "ready"
    content: str
    answer_kind: Literal["fact", "bank_position", "ai_inference"] | None = None
    sources: list[ChatSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    thread: ChatThreadResponse
    messages: list[ChatMessageResponse]
    limit: int
    has_more: bool
    next_before: UUID | None = None


class ChatMessageRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    client_message_id: UUID


class ChatMessageResult(BaseModel):
    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse
    reused: bool = False


class ChatAnswerClaim(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    kind: Literal["fact", "bank_position", "ai_inference"]
    evidence_refs: list[str] = Field(min_length=1, max_length=8)


class ChatAnswerPayload(BaseModel):
    answer: str = Field(min_length=1, max_length=12_000)
    claims: list[ChatAnswerClaim] = Field(default_factory=list, max_length=12)
    missing_evidence: list[str] = Field(default_factory=list, max_length=12)
