import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit

import httpx2 as httpx

from app.core.config import Settings
from app.services.rss.catalog import FeedConfig


class FetchError(Exception):
    def __init__(self, message: str, status: int | None = None, retry_after: int = 0):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True)
class FetchResult:
    status: int
    body: bytes
    etag: str | None
    last_modified: str | None


def retry_after_seconds(value: str | None) -> int:
    if not value:
        return 0
    try:
        seconds = int(value)
    except ValueError:
        try:
            seconds = int((parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0
    return max(0, seconds)


async def validate_destination(url: str, config: FeedConfig) -> None:
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname not in config.allowed_hosts
        or parts.port not in {None, 443}
        or parts.username is not None
        or parts.password is not None
    ):
        raise FetchError("Feed destination is not allowed")
    addresses = await asyncio.get_running_loop().getaddrinfo(
        parts.hostname, 443, type=socket.SOCK_STREAM
    )
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise FetchError("Feed destination must use public IP addresses")


async def fetch_feed(
    client: httpx.AsyncClient,
    config: FeedConfig,
    settings: Settings,
    etag: str | None = None,
    last_modified: str | None = None,
) -> FetchResult:
    headers = {}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    url = config.url
    async with asyncio.timeout(settings.rss_request_timeout_seconds):
        for _ in range(4):
            await validate_destination(url, config)
            async with client.stream(
                "GET", url, headers=headers, follow_redirects=False
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise FetchError("Redirect without Location", response.status_code)
                    url = urljoin(url, location)
                    continue
                if response.status_code not in {200, 304}:
                    raise FetchError(
                        f"Feed returned HTTP {response.status_code}",
                        response.status_code,
                        retry_after_seconds(response.headers.get("retry-after")),
                    )
                body = bytearray()
                if response.status_code == 200:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        body.extend(chunk)
                        if len(body) > settings.rss_max_response_bytes:
                            raise FetchError("Feed exceeds decompressed size limit", 200)
                return FetchResult(
                    response.status_code,
                    bytes(body),
                    response.headers.get("etag"),
                    response.headers.get("last-modified"),
                )
    raise FetchError("Too many feed redirects")
