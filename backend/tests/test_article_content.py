import asyncio
import json
from unittest.mock import AsyncMock

import httpx2 as httpx
import pytest

from app.core.config import Settings
from app.services.rss.article_content import (
    ArticleTextUnavailable,
    extract_article_text,
    fetch_article_text,
)
from app.services.rss.catalog import load_catalog
from app.services.rss.fetcher import FetchError
from app.services.rss.parser import parse_feed
from tests.test_rss import item, rss

BODY = (
    "Компания открыла новый завод в регионе. Производство начнёт работу в следующем году. "
    "На предприятии будут выпускать оборудование для местных поставщиков и создавать рабочие места."
)


@pytest.mark.parametrize(("source", "container"), [
    ("rbc", '<div class="article__text">'),
    ("kommersant", '<div class="doc-text-block__paragraph">'),
    ("vedomosti", '<div class="box-paragraph__text">'),
    ("bfm", '<div class="current-article js-mediator-article">'),
    ("klerk", '<section class="articleStyles paywall">'),
])
def test_publisher_body_excludes_caption_navigation_and_recommendations(source, container):
    end = "</section>" if source == "klerk" else "</div>"
    html = (
        f"<nav>Меню</nav><figure><figcaption>Джефф Безос</figcaption></figure>{container}"
        f"<p>{BODY}</p><script>bad()</script><div class='more_articles'>Реклама</div>{end}"
        "<footer>Служебный текст</footer>"
    )
    assert extract_article_text(html, source) == BODY


def test_jsonld_graph_article_body_and_access_restriction():
    node = {"@type": "NewsArticle", "articleBody": BODY, "isAccessibleForFree": True}
    html = '<script type="application/ld+json">' + json.dumps({"@graph": [node]}) + "</script>"
    assert extract_article_text(html, "bfm") == BODY
    node["isAccessibleForFree"] = False
    html = '<script type="application/ld+json">' + json.dumps(node) + "</script>"
    with pytest.raises(ArticleTextUnavailable, match="subscription"):
        extract_article_text(html, "bfm")


def test_caption_or_whole_page_is_not_a_body():
    with pytest.raises(ArticleTextUnavailable):
        extract_article_text(f"<body><nav>{BODY}</nav><p>Джефф Безос</p></body>", "bfm")
    with pytest.raises(ArticleTextUnavailable):
        extract_article_text('<div class="current-article">Джефф Безос</div>', "bfm")


def test_rbc_missing_full_text_does_not_use_photo_caption():
    config = next(c for c in load_catalog() if c.code == "rbc")
    body = rss(item(extra=(
        '<content:encoded xmlns:content="http://purl.org/rss/1.0/modules/content/">'
        '<![CDATA[<p>Джефф Безос</p>]]></content:encoded>'
    )))
    assert parse_feed(body, config).articles[0].content_text is None


def test_article_fetch_redirect_host_checks_and_size_limit(monkeypatch):
    config = next(c for c in load_catalog() if c.code == "bfm")
    validation = AsyncMock()
    monkeypatch.setattr("app.services.rss.article_content.validate_destination", validation)

    async def scenario():
        def handler(request):
            if request.url.path == "/start":
                return httpx.Response(302, headers={"Location": "/news/1"})
            return httpx.Response(200, headers={"Content-Type": "text/html; charset=utf-8"},
                                  text=f'<div class="current-article"><p>{BODY}</p></div>')
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await fetch_article_text(
                client, "https://www.bfm.ru/start", config, Settings(_env_file=None)
            ) == BODY
            assert validation.await_count == 2
            with pytest.raises(FetchError, match="size limit"):
                await _oversized(config)
    # Use a separate response so the test does not depend on the body's character encoding.
    async def _oversized(config):
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, headers={"Content-Type": "text/html"},
                                          content=b"x" * 1025)
        )) as big_client:
            await fetch_article_text(big_client, "https://www.bfm.ru/news/1", config,
                                     Settings(_env_file=None, rss_max_response_bytes=1024))
    asyncio.run(scenario())


def test_article_redirect_outside_publisher_is_rejected(monkeypatch):
    config = next(c for c in load_catalog() if c.code == "bfm")
    # Only mock DNS; the real host/IP restrictions still run on every redirect.
    async def scenario():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "getaddrinfo", AsyncMock(return_value=[
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ]))
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(302, headers={"Location": "https://127.0.0.1/"})
        )) as client:
            with pytest.raises(FetchError, match="not allowed"):
                await fetch_article_text(client, "https://www.bfm.ru/news/1", config,
                                         Settings(_env_file=None))
    asyncio.run(scenario())
