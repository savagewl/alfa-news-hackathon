"""Extract article bodies from publisher HTML; never use the entire page as text."""

import asyncio
import json
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx2 as httpx

from app.core.config import Settings
from app.services.rss.catalog import FeedConfig
from app.services.rss.fetcher import FetchError, validate_destination
from app.services.rss.parser import plain_text

BODY_CLASSES = {
    "rbc": {"article__text"},
    "kommersant": {"doc-text-block__paragraph", "b-article__text"},
    "vedomosti": {"box-paragraph__text"},
    "bfm": {"current-article"},
    "klerk": {"articleStyles"},
}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
             "param", "source", "track", "wbr"}
SKIP_TAGS = {"script", "style", "noscript", "nav", "aside", "footer", "header", "form",
             "figure", "figcaption", "button", "svg"}
SKIP_CLASSES = {"about-article", "more_articles", "banner", "adfox", "advertisement",
                "article-recommendation", "article-share"}
ARTICLE_TYPES = {"Article", "NewsArticle", "ReportageNewsArticle", "BlogPosting"}


class ArticleTextUnavailable(ValueError):
    pass


class ArticleHTMLParser(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.selectors = BODY_CLASSES[source]
        self.stack: list[tuple[str, bool, bool]] = []
        self.parts: list[str] = []
        self.json_blocks: list[str] = []
        self.json_parts: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        parent_selected, parent_hidden = self.stack[-1][1:] if self.stack else (False, False)
        hidden = parent_hidden or tag in SKIP_TAGS or bool(classes & SKIP_CLASSES)
        hidden = hidden or "hidden" in attrs or attrs.get("aria-hidden") == "true"
        selected = parent_selected or bool(classes & self.selectors)
        if selected and not hidden and tag in {"p", "div", "li", "br", "blockquote"}:
            self.parts.append("\n")
        if tag == "script" and attrs.get("type", "").lower() == "application/ld+json":
            self.json_parts = []
        if tag not in VOID_TAGS:
            self.stack.append((tag, selected, hidden))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "script" and self.json_parts is not None:
            self.json_blocks.append("".join(self.json_parts))
            self.json_parts = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                if self.stack[index][1] and not self.stack[index][2]:
                    if tag in {"p", "div", "li", "blockquote"}:
                        self.parts.append("\n")
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self.json_parts is not None:
            self.json_parts.append(data)
        elif self.stack and self.stack[-1][1] and not self.stack[-1][2]:
            self.parts.append(data)


def _article_nodes(value):
    pending = [value]
    while pending:
        node = pending.pop()
        if isinstance(node, list):
            pending.extend(node)
        elif isinstance(node, dict):
            types = node.get("@type", [])
            types = [types] if isinstance(types, str) else types
            if isinstance(types, list) and ARTICLE_TYPES.intersection(
                item for item in types if isinstance(item, str)
            ):
                yield node
            graph = node.get("@graph")
            if isinstance(graph, (list, dict)):
                pending.append(graph)


def extract_article_text(html: str, source: str) -> str:
    parser = ArticleHTMLParser(source)
    parser.feed(html)
    structured = []
    for block in parser.json_blocks:
        try:
            nodes = list(_article_nodes(json.loads(block)))
        except (ValueError, RecursionError):
            continue
        for node in nodes:
            if node.get("isAccessibleForFree") in (False, "false"):
                raise ArticleTextUnavailable("Article requires a subscription")
            if isinstance(node.get("articleBody"), str):
                structured.append(plain_text(node["articleBody"]) or "")
    # Visible article containers avoid captions and navigation; JSON-LD is a fallback.
    visible = "\n\n".join(
        line for part in "".join(parser.parts).splitlines() if (line := " ".join(part.split()))
    )
    candidates = [visible, *structured]
    for text in candidates:
        if len(text) >= 80 and len(text.split()) >= 12:
            return text
    raise ArticleTextUnavailable("Article body not found; summary/caption is not a full text")


async def fetch_article_text(
    client: httpx.AsyncClient, url: str, config: FeedConfig, settings: Settings
) -> str:
    # Article hosts differ from RSS export hosts. Each redirect is checked again.
    host = urlsplit(config.site_url).hostname
    hosts = {host, host.removeprefix("www.")}
    article_config = config.model_copy(update={"allowed_hosts": sorted(hosts)})
    async with asyncio.timeout(settings.rss_request_timeout_seconds):
        for _ in range(4):
            await validate_destination(url, article_config)
            async with client.stream(
                "GET", url, follow_redirects=False,
                headers={"Accept": "text/html,application/xhtml+xml"},
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise FetchError("Article redirect without Location", response.status_code)
                    url = urljoin(url, location)
                    continue
                if response.status_code != 200:
                    raise FetchError(f"Article returned HTTP {response.status_code}",
                                     response.status_code)
                if not any(t in response.headers.get("content-type", "").lower()
                           for t in ("text/html", "application/xhtml+xml")):
                    raise FetchError("Article response is not HTML", 200)
                body = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    body.extend(chunk)
                    if len(body) > settings.rss_max_response_bytes:
                        raise FetchError("Article exceeds decompressed size limit", 200)
                html = bytes(body).decode(response.encoding or "utf-8", errors="replace")
            return await asyncio.to_thread(extract_article_text, html, config.code)
    raise FetchError("Too many article redirects")
