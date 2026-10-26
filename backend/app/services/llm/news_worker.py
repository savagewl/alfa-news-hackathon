"""In-process, PostgreSQL-leased worker for news AI preprocessing."""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models.news_embeddings import EMBEDDING_DIM
from app.repositories.news_embeddings import (
    NewsClaim,
    NewsLeaseLostError,
    claim_news_batch,
    mark_news_claim_failed,
    renew_news_lease,
    store_news_claim,
)
from app.services.llm.config import ServicesSettings
from app.services.llm.embeddings import EmbeddingsClient
from app.services.llm.news_analysis import process_news_text
from app.services.llm.provider import (
    LLMBudgetExceededError,
    LLMConfigError,
    LLMProvider,
    get_llm_provider,
)

logger = logging.getLogger(__name__)


class NewsAIWorker:
    def __init__(self, factory: async_sessionmaker, settings: ServicesSettings):
        self.factory = factory
        self.settings = settings
        self.stop_event = asyncio.Event()
        self.task: asyncio.Task | None = None
        self._embedder: EmbeddingsClient | None = None
        self._owns_embedder = False
        self._transient_failure = False
        self._budget_pause_seconds = 0.0

    def start(self) -> None:
        self.task = asyncio.create_task(self.run(), name="news-ai-supervisor")

    async def stop(self) -> None:
        self.stop_event.set()
        if self.task is None:
            return
        try:
            async with asyncio.timeout(10):
                await asyncio.shield(self.task)
        except TimeoutError:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        if self._owns_embedder and self._embedder is not None:
            await asyncio.to_thread(self._embedder.close)
            self._embedder = None

    async def pause(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                await self.stop_event.wait()

    async def heartbeat(self, claim: NewsClaim) -> None:
        try:
            while True:
                await asyncio.sleep(self.settings.news_worker_lease_seconds / 3)
                await renew_news_lease(
                    self.factory, claim, self.settings.news_worker_lease_seconds
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # The final write is still conditional on the token. If renewal was
            # lost, a stale network result is discarded instead of overwriting.
            logger.warning(
                "News AI lease renewal stopped article=%s type=%s",
                claim.article_id,
                type(exc).__name__,
            )

    async def process(
        self,
        claim: NewsClaim,
        llm: LLMProvider,
        embedder: EmbeddingsClient,
    ) -> None:
        heartbeat = asyncio.create_task(self.heartbeat(claim))
        try:
            # Claims contain immutable inputs; no DB session/transaction is held
            # during the blocking LLM and Ollama requests.
            processed = await asyncio.to_thread(
                process_news_text,
                claim.text,
                claim.url,
                llm=llm,
                embedder=embedder,
                settings=self.settings,
            )
            if len(processed.vector) != EMBEDDING_DIM:
                raise ValueError(
                    f"Размерность вектора {len(processed.vector)} != {EMBEDDING_DIM}"
                )
            await store_news_claim(self.factory, claim, processed, self.settings, llm)
        except NewsLeaseLostError:
            logger.warning(
                "News AI lease lost article=%s; stale result discarded", claim.article_id
            )
        except LLMBudgetExceededError as exc:
            await mark_news_claim_failed(
                self.factory, claim, str(exc), self.settings, budget_blocked=True
            )
            logger.warning("News AI daily token budget exhausted; work deferred until next UTC day")
            now = datetime.now(UTC)
            next_day = datetime.combine(
                (now + timedelta(days=1)).date(), datetime.min.time(), tzinfo=UTC
            )
            self._budget_pause_seconds = max((next_day - now).total_seconds(), 1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._transient_failure = True
            logger.warning(
                "News AI processing failed article=%s type=%s",
                claim.article_id,
                type(exc).__name__,
            )
            with contextlib.suppress(NewsLeaseLostError):
                await mark_news_claim_failed(
                    self.factory,
                    claim,
                    f"{type(exc).__name__}: {exc}",
                    self.settings,
                )
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError, NewsLeaseLostError):
                await heartbeat

    async def run_once(
        self,
        llm: LLMProvider | None = None,
        embedder: EmbeddingsClient | None = None,
    ) -> int:
        self._transient_failure = False
        self._budget_pause_seconds = 0.0
        llm = llm or get_llm_provider(self.settings)
        if embedder is None:
            if self._embedder is None:
                self._embedder = EmbeddingsClient(self.settings)
                self._owns_embedder = True
            embedder = self._embedder
        claims = await claim_news_batch(
            self.factory, self.settings, llm=llm, embedder=embedder
        )
        if not claims:
            return 0
        semaphore = asyncio.Semaphore(self.settings.news_worker_concurrency)

        async def bounded(claim: NewsClaim) -> None:
            async with semaphore:
                await self.process(claim, llm, embedder)

        async with asyncio.TaskGroup() as group:
            for claim in claims:
                group.create_task(bounded(claim), name=f"news-ai-{claim.article_id}")
        return len(claims)

    async def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                count = await self.run_once()
            except LLMConfigError as exc:
                # Configuration absence is operational, not an API startup failure.
                logger.error("News AI worker paused: %s", exc)
                await self.pause(self.settings.news_worker_config_retry_seconds)
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("News AI worker iteration failed")
                await self.pause(self.settings.news_worker_retry_seconds)
                continue
            if self._budget_pause_seconds:
                await self.pause(self._budget_pause_seconds)
            elif self._transient_failure:
                # A missing Ollama/provider endpoint should not hammer every
                # accumulated article in a tight loop.
                await self.pause(self.settings.news_worker_retry_seconds)
            elif count == 0:
                await self.pause(self.settings.news_worker_poll_seconds)
