"""Background preparation of company profile vectors."""

import asyncio
import logging
import math

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.company import Company
from app.repositories.company_profiles import (
    claim_company_profile,
    complete_company_profile,
    fail_company_profile,
    prepare_company_profile_input,
)
from app.services.llm.company_profile import embed_company_profile
from app.services.llm.config import ServicesSettings
from app.services.llm.embeddings import EmbeddingsClient

logger = logging.getLogger("app.services.company_profiles")


class CompanyProfileWorker:
    """Poll companies and refresh stale profile vectors outside request handling."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        settings: ServicesSettings,
        *,
        embedder: EmbeddingsClient | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings
        self._embedder = embedder
        self._owns_embedder = embedder is None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._after_company_id: str | None = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="company-profile-worker")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except TimeoutError:
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._owns_embedder and self._embedder is not None:
            close = getattr(self._embedder, "close", None)
            if close is None:
                close = getattr(getattr(self._embedder, "_client", None), "close", None)
            if close is not None:
                close()
            self._embedder = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                processed = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Company profile polling iteration failed")
                processed = 0
            delay = 0 if processed else self._settings.profile_poll_seconds
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def run_once(self) -> int:
        if not self._settings.profile_enabled:
            return 0
        async with self._session_factory() as session:
            query = select(Company).order_by(Company.id).limit(self._settings.profile_batch_size)
            if self._after_company_id is not None:
                query = query.where(Company.id > self._after_company_id)
            companies = list(await session.scalars(query))
            if not companies and self._after_company_id is not None:
                self._after_company_id = None
                return 0
            if companies:
                self._after_company_id = companies[-1].id

        claimed = 0
        for company in companies:
            token = None
            prepared = None
            async with self._session_factory() as session:
                try:
                    # Reload in this session; the scan session has already been released.
                    current = await session.get(Company, company.id)
                    if current is None:
                        continue
                    prepared = await prepare_company_profile_input(session, current, self._settings)
                    token = await claim_company_profile(
                        session,
                        prepared,
                        lease_seconds=max(
                            self._settings.profile_lease_seconds,
                            math.ceil(self._settings.embed_timeout_seconds) + 30,
                        ),
                    )
                    if token is None:
                        continue
                    claimed += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("Company profile failed company=%s: %s", company.id, exc)
                    continue
            if token is None or prepared is None:
                continue
            try:
                # No database session or transaction is held during the network call.
                if self._embedder is None:
                    self._embedder = EmbeddingsClient(self._settings)
                vectors = await asyncio.to_thread(
                    embed_company_profile,
                    prepared.text,
                    embedder=self._embedder,
                    settings=self._settings,
                )
                expected_keys = {"main", *prepared.text.aspects}
                if set(vectors) != expected_keys:
                    raise ValueError("Company profile embedding is incomplete")
                if any(
                    len(vector) != self._settings.embed_dim
                    or not all(math.isfinite(value) for value in vector)
                    for vector in vectors.values()
                ):
                    raise ValueError("Company profile embedding has invalid values or dimension")
                async with self._session_factory() as session:
                    await complete_company_profile(session, prepared, token, vectors)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Company profile failed company=%s: %s", company.id, exc)
                async with self._session_factory() as session:
                    await fail_company_profile(
                        session,
                        company.id,
                        token,
                        retry_seconds=self._settings.profile_retry_seconds,
                        error=type(exc).__name__,
                    )
        return claimed
