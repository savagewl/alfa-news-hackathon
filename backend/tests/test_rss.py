import asyncio
import base64
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx2 as httpx
import pytest
from defusedxml.common import DefusedXmlException
from fastapi.testclient import TestClient

from app.api.news import decode_cursor
from app.core.config import Settings
from app.main import create_app
from app.services.rss.catalog import load_catalog
from app.services.rss.fetcher import FetchError, fetch_feed, retry_after_seconds
from app.services.rss.parser import canonical_url, parse_feed, plain_text


def rss(items: str) -> bytes:
    return (
        '<rss version="2.0" xmlns:rbc="https://www.rbc.ru"><channel>'
        "<title>Test</title><link>https://example.com</link><description>Test</description>"
        f"{items}</channel></rss>"
    ).encode()


def item(title="News", link="https://example.com/article", extra=""):
    return f"<item><title>{title}</title><link>{link}</link>{extra}</item>"


@pytest.mark.parametrize("prefix", ["rbc", "rbc_news"])
def test_five_sources_and_source_specific_fields(prefix):
    configs = load_catalog()
    assert {c.code for c in configs} == {"rbc", "kommersant", "vedomosti", "bfm", "klerk"}
    rbc = next(c for c in configs if c.code == "rbc")
    parsed = parse_feed(
        rss(
            item(
                extra=(
                    "<guid>guid-1</guid><rbc:news_id>123</rbc:news_id>"
                    "<rbc:newsModifDate>2026-10-21T12:00:00+03:00</rbc:newsModifDate>"
                    "<rbc:anons>Описание</rbc:anons><rbc:full-text>Полный текст</rbc:full-text>"
                    "<content:encoded xmlns:content='http://purl.org/rss/1.0/modules/content/'>"
                    "<![CDATA[<p>Джефф Безос</p>]]></content:encoded>"
                    "<category>Экономика</category>"
                    "<pubDate>Wed, 21 Oct 2026 11:00:00 +0300</pubDate>"
                )
            )
        ).replace(b"rbc:", prefix.encode() + b":").replace(
            b"xmlns:rbc=", b"xmlns:" + prefix.encode() + b"="
        ),
        rbc,
    )
    assert not parsed.errors
    article = parsed.articles[0]
    assert article.external_id == "123"
    assert article.summary == "Описание"
    assert article.content_text == "Полный текст"  # в каталоге store_content=true
    assert article.source_updated_at == datetime(2026, 10, 21, 9, tzinfo=UTC)
    assert article.published_at == datetime(2026, 10, 21, 8, tzinfo=UTC)
    assert article.topics == ["economy"]
    without_content = parse_feed(
        rss(item(extra="<rbc:full-text>Полный текст</rbc:full-text>")),
        rbc.model_copy(update={"store_content": False}),
    )
    assert without_content.articles[0].content_text is None


@pytest.mark.parametrize("code", ["vedomosti", "bfm", "kommersant", "klerk"])
def test_sparse_entries_and_bad_item_isolation(code):
    config = next(c for c in load_catalog() if c.code == code)
    parsed = parse_feed(rss(item() + "<item><title>Missing URL</title></item>"), config)
    assert len(parsed.articles) == 1
    article = parsed.articles[0]
    assert article.summary is None
    assert article.external_id is None
    assert article.published_at is None
    assert len(parsed.errors) == 1
    assert parsed.errors[0].index == 1


def test_atom_html_and_timezone():
    body = b"""<feed xmlns="http://www.w3.org/2005/Atom"><title>Test</title>
    <entry><id>tag:entry</id><title>Atom</title><link href="https://example.com/a"/>
    <summary type="html">&lt;p&gt;Hello&lt;/p&gt;&lt;script&gt;bad()&lt;/script&gt;</summary>
    <published>2026-10-21T12:00:00</published><author><name>Author</name></author>
    </entry></feed>"""
    article = parse_feed(body, load_catalog()[0]).articles[0]
    assert article.summary == "Hello"
    assert article.authors == ["Author"]
    assert article.published_at == datetime(2026, 10, 21, 9, tzinfo=UTC)


@pytest.mark.parametrize("body", [b"<html><body>Welcome</body></html>", b"<rss><channel>"])
def test_invalid_feed_rejected(body):
    with pytest.raises((ValueError, SyntaxError)):
        parse_feed(body, load_catalog()[0])


def test_external_entities_rejected():
    body = b'<!DOCTYPE rss [<!ENTITY test SYSTEM "file:///etc/passwd">]><rss>&test;</rss>'
    with pytest.raises(DefusedXmlException):
        parse_feed(body, load_catalog()[0])


def test_url_normalization_and_text_safety():
    assert canonical_url("https://Example.com/A?id=1&amp=2&utm_source=x#top", "") == (
        "https://example.com/A?id=1&amp=2"
    )
    assert plain_text("<p>A &amp; B</p><script>secret()</script><p>C</p>") == "A & B C"
    with pytest.raises(ValueError):
        canonical_url("javascript:alert(1)", "")


def test_invalid_cursor():
    from fastapi import HTTPException

    for value in ("bad", "e30", "W10", "!", "bnVsbA"):
        with pytest.raises(HTTPException) as exc:
            decode_cursor(value)
        assert exc.value.status_code == 422
    invalid = base64.urlsafe_b64encode(json.dumps(["2026-10-21T12:00:00Z", 1]).encode()).decode()
    with pytest.raises(HTTPException) as exc:
        decode_cursor(invalid)
    assert exc.value.status_code == 422


def test_fetch_conditional_request_redirect_and_size_limit(monkeypatch):
    monkeypatch.setattr("app.services.rss.fetcher.validate_destination", AsyncMock())

    async def scenario():
        config = load_catalog()[0]
        settings = Settings(_env_file=None, rss_max_response_bytes=1024)
        seen = []

        def handler(request):
            seen.append(request)
            if request.url.path.endswith("full.rss"):
                return httpx.Response(302, headers={"location": "/redirected"})
            return httpx.Response(304, headers={"etag": '"new"'})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await fetch_feed(client, config, settings, '"old"', "date")
        assert result.status == 304
        assert result.body == b""
        assert len(seen) == 2
        assert seen[0].headers["if-none-match"] == '"old"'
        assert seen[0].headers["if-modified-since"] == "date"
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"a" * 1025))
        ) as client:
            with pytest.raises(FetchError, match="size limit"):
                await fetch_feed(client, config, settings)

    asyncio.run(scenario())


def test_fetch_retry_after(monkeypatch):
    monkeypatch.setattr("app.services.rss.fetcher.validate_destination", AsyncMock())

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(429, headers={"retry-after": "123"})
            )
        ) as client:
            with pytest.raises(FetchError) as exc:
                await fetch_feed(client, load_catalog()[0], Settings(_env_file=None))
            assert exc.value.status == 429
            assert exc.value.retry_after == 123

    asyncio.run(scenario())
    assert retry_after_seconds("garbage") == 0


def test_redirect_to_private_or_unlisted_host_rejected():
    from app.services.rss.fetcher import validate_destination

    for url in ("http://rssexport.rbc.ru/rss", "https://127.0.0.1/", "https://other.test/"):
        with pytest.raises(FetchError):
            asyncio.run(validate_destination(url, load_catalog()[0]))


def test_worker_lifecycle_and_news_auth(monkeypatch):
    from unittest.mock import Mock

    stop = AsyncMock()
    started = Mock()
    monkeypatch.setattr("app.main.RssWorker.start", started)
    monkeypatch.setattr("app.main.RssWorker.stop", stop)
    app = create_app(Settings(_env_file=None, rss_enabled=True))
    with TestClient(app) as client:
        started.assert_called_once()
        assert client.get("/health").status_code == 200
        for path in ("/news", "/news/sources"):
            assert client.get(path).status_code == 401
        schema = client.get("/openapi.json").json()
        assert "/news/{news_id}" in schema["paths"]
    stop.assert_awaited_once()
