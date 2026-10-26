import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import SessionDep

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(default="ok", description="API работает.")


class ReadinessResponse(HealthResponse):
    database: Literal["ok"] = Field(default="ok", description="PostgreSQL доступен.")


class UnavailableResponse(BaseModel):
    detail: str = Field(default="Database unavailable", description="Причина недоступности.")


@router.get(
    "",
    response_model=HealthResponse,
    summary="Проверить работоспособность API",
    description="Возвращает 200, если API принимает запросы. Подключение к БД не проверяется.",
    operation_id="checkHealth",
    response_description="API работает",
)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Проверить готовность API и PostgreSQL",
    description=(
        "Выполняет `SELECT 1` в PostgreSQL с таймаутом 5 секунд. "
        "Возвращает 200 при доступной БД или 503, если подключение не удалось."
    ),
    operation_id="checkReadiness",
    response_description="API и PostgreSQL готовы к работе",
    responses={
        503: {
            "model": UnavailableResponse,
            "description": "PostgreSQL недоступен или превышен таймаут проверки",
            "content": {"application/json": {"example": {"detail": "Database unavailable"}}},
        }
    },
)
async def readiness(session: SessionDep) -> ReadinessResponse:
    try:
        async with asyncio.timeout(5):
            await session.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError, TimeoutError):
        logger.warning("Database readiness check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from None
    return ReadinessResponse()
