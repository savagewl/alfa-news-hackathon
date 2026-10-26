import asyncio
import contextlib
import logging
import time
import zlib
from urllib.parse import urlsplit

import httpx2 as httpx
from defusedxml.common import DefusedXmlException
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.repositories.news import (
    FeedClaim,
    LeaseLostError,
    claim_feed,
    cleanup_history,
    finish_failure,
    finish_success,
    renew_lease,
    save_snapshot,
    sync_catalog,
)
from app.services.rss.catalog import FeedConfig, load_catalog
from app.services.rss.fetcher import FetchError, FetchResult, fetch_feed
from app.services.rss.parser import parse_feed

logger = logging.getLogger(__name__)


class RssWorker:
    def __init__(self, factory: async_sessionmaker, settings: Settings):
        self.factory = factory
        self.settings = settings
        self.configs = load_catalog()
        self.stop_event = asyncio.Event()
        self.task: asyncio.Task | None = None
        self.domain_locks: dict[str, asyncio.Lock] = {}

    def start(self) -> None:
        self.task = asyncio.create_task(self.run(), name="rss-supervisor")

    async def stop(self) -> None:
        self.stop_event.set()
        if self.task is None:
            return
        try:
            async with asyncio.timeout(35):
                await asyncio.shield(self.task)
        except TimeoutError:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    async def pause(self, seconds: float) -> None:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(seconds):
                await self.stop_event.wait()

    async def run(self) -> None:
        # Database failures must not kill the background task or prevent API startup.
        while not self.stop_event.is_set():
            try:
                await sync_catalog(self.factory, self.configs)
                async with httpx.AsyncClient(
                    timeout=self.settings.rss_request_timeout_seconds,
                    limits=httpx.Limits(max_connections=self.settings.rss_concurrency),
                    headers={
                        "User-Agent": "AlfaNewsRSS/0.1",
                        "Accept": (
                            "application/rss+xml, application/atom+xml, "
                            "application/xml, text/xml, */*;q=0.1"
                        ),
                    },
                    trust_env=False,
                ) as client:
                    async with asyncio.TaskGroup() as group:
                        for index in range(self.settings.rss_concurrency):
                            group.create_task(self.poll_loop(client), name=f"rss-worker-{index}")
                        group.create_task(self.maintenance_loop(), name="rss-maintenance")
                return
            except Exception:
                logger.exception("RSS supervisor failed; restarting in 10 seconds")
                await self.pause(10)

    async def maintenance_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await cleanup_history(self.factory, self.settings)
            except Exception:
                logger.exception("RSS history cleanup failed")
            await self.pause(3600)

    async def poll_loop(self, client: httpx.AsyncClient) -> None:
        configs = {c.feed_code: c for c in self.configs}
        while not self.stop_event.is_set():
            try:
                claim = await claim_feed(
                    self.factory, self.configs, self.settings.rss_lease_seconds
                )
                if claim is None:
                    await self.pause(2)
                    continue
                await self.process(client, claim, configs[claim.feed_code])
            except LeaseLostError:
                logger.warning("RSS lease lost; stale worker result discarded")
            except Exception:
                # Leave a durable pending snapshot and expiring lease for recovery.
                logger.exception("RSS worker failed; lease will allow recovery")
                await self.pause(5)

    async def heartbeat(self, claim: FeedClaim) -> None:
        try:
            while True:
                await asyncio.sleep(self.settings.rss_lease_seconds / 3)
                await renew_lease(self.factory, claim, self.settings.rss_lease_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("RSS lease renewal failed")

    async def process(
        self, client: httpx.AsyncClient, claim: FeedClaim, config: FeedConfig
    ) -> None:
        heartbeat = asyncio.create_task(self.heartbeat(claim))
        started = time.monotonic()
        try:
            result = None
            try:
                if claim.pending_body is not None:
                    body = await asyncio.to_thread(zlib.decompress, claim.pending_body)
                    result = FetchResult(200, body, claim.pending_etag, claim.pending_last_modified)
                else:
                    host = urlsplit(config.url).hostname
                    lock = self.domain_locks.setdefault(host, asyncio.Lock())
                    async with lock:
                        result = await fetch_feed(
                            client, config, self.settings, claim.etag, claim.last_modified
                        )
                    if result.status == 200:
                        await save_snapshot(self.factory, claim, result)
                parsed = None
                if result.status == 200:
                    parsed = await asyncio.to_thread(parse_feed, result.body, config)
            except (
                FetchError,
                httpx.HTTPError,
                OSError,
                TimeoutError,
                ValueError,
                SyntaxError,
                DefusedXmlException,
            ) as exc:
                if isinstance(exc, FetchError):
                    error, status, retry = str(exc), exc.status, exc.retry_after
                else:
                    error = f"RSS {type(exc).__name__}"
                    status, retry = result.status if result else None, 0
                await finish_failure(self.factory, claim, config, error, status, retry)
                logger.warning("RSS feed=%s error=%s", config.feed_code, error)
                return
            counts = await finish_success(self.factory, claim, config, result, parsed)
            logger.info(
                "RSS feed=%s status=%s duration=%.2f counts=%s",
                config.feed_code,
                result.status,
                time.monotonic() - started,
                counts,
            )
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
