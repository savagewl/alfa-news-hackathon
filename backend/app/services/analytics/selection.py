from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analytics import BankAnalyticsSelection
from app.services.llm.config import ServicesSettings


async def effective_bank_settings(
    session: AsyncSession, settings: ServicesSettings
) -> ServicesSettings:
    selection = await session.get(BankAnalyticsSelection, 1)
    if selection is None:
        return settings
    return settings.model_copy(
        update={"bank_document_ids": [UUID(value) for value in selection.document_ids]}
    )
