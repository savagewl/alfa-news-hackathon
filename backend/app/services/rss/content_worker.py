"""Background fetching of bodies missing from RSS, including the existing corpus."""

import asyncio
import contextlib
import logging

import httpx2 as httpx
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.repositories.article_content import claim_article_content, finish_article_content
from app.services.rss.article_content import fetch_article_text
from app.services.rss.catalog import load_catalog

logger = logging.getLogger(__name__)


class ArticleContentWorker:
    def __init__(self, factory: async_sessionmaker, settings: Settings):
        self.factory = factory
        self.settings = settings
        self.configs = {c.code: c for c in load_catalog() if c.store_content}
        self.stop_event = asyncio.Event()
        self.task: asyncio.Task | None = None
        self.domain_locks = {code: asyncio.Lock() for code in self.configs}

    def start(self) -> None:
        self.task = asyncio.create_task(self.run(), name="article-content-supervisor")

    async def stop(self) -> None:
        self.stop_event.set()
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    async def pause(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                await self.stop_event.wait()

    async def process_once(self, client: httpx.AsyncClient) -> bool:
        claim = await claim_article_content(self.factory, list(self.configs))
        if claim is None:
            return False
        try:
            async with self.domain_locks[claim.source_code]:
                text = await fetch_article_text(
                    client, claim.url, self.configs[claim.source_code], self.settings
                )
                await self.pause(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Article body failed article=%s type=%s",
                           claim.article_id, type(exc).__name__)
            await finish_article_content(self.factory, claim, error=f"{type(exc).__name__}: {exc}")
        else:
            await finish_article_content(self.factory, claim, text=text)
        return True

    async def loop(self, client: httpx.AsyncClient) -> None:
        while not self.stop_event.is_set():
            try:
                processed = await self.process_once(client)
            except Exception:
                logger.exception("Article body worker iteration failed")
                await self.pause(10)
            else:
                if not processed:
                    await self.pause(self.settings.article_content_poll_seconds)

    async def run(self) -> None:
        async with httpx.AsyncClient(
            timeout=self.settings.rss_request_timeout_seconds, trust_env=False,
            headers={"User-Agent": "AlfaNewsRSS/0.1"},
        ) as client:
            async with asyncio.TaskGroup() as group:
                for _ in range(self.settings.article_content_concurrency):
                    group.create_task(self.loop(client))
