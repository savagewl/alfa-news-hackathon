import calendar
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import feedparser
from defusedxml.ElementTree import fromstring

from app.services.rss.catalog import FeedConfig

NORMALIZER_VERSION = 2
TOPIC_RULES = {
    "economy": ("эконом", "макроэконом"),
    "politics": ("полит", "власть", "международ"),
    "business": ("бизнес", "компани", "предприним", "отрасл", "торговл"),
    "finance": ("финанс", "банк", "инвест", "рынки", "бирж"),
    "taxes": ("налог", "бухгалтер", "отчет", "отчёт"),
    "regulation": ("закон", "регулир", "право"),
    "technology": ("технолог", "интернет", "цифров", "телеком"),
}


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"p", "br", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {"p", "div", "li"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def plain_text(value: str | None) -> str | None:
    parser = TextExtractor()
    parser.feed(value or "")
    return " ".join("".join(parser.parts).split()) or None


def canonical_url(value: str, base: str) -> str:
    parts = urlsplit(urljoin(base, value.strip()))
    if (
        parts.scheme not in {"https", "http"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
    ):
        raise ValueError("Invalid article URL")
    # Keep functional query parameters, path case, trailing slash and scheme.
    query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"yclid", "gclid", "fbclid"}
    ]
    port = parts.port
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port and (parts.scheme, port) not in {("https", 443), ("http", 80)}:
        host = f"{host}:{port}"
    return urlunsplit((parts.scheme, host, parts.path or "/", urlencode(query), ""))


def parse_date(value, fallback, timezone: str) -> datetime | None:
    if isinstance(value, str) and value.strip():
        try:
            parsed = parsedate_to_datetime(value)
        except (ValueError, TypeError, OverflowError):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=ZoneInfo(timezone))
            return parsed.astimezone(UTC)
    if fallback:
        try:
            return datetime.fromtimestamp(calendar.timegm(fallback), UTC)
        except (ValueError, OverflowError, TypeError):
            pass
    return None


@dataclass(frozen=True)
class ParsedArticle:
    external_id: str | None
    original_url: str
    canonical_url: str
    title: str
    summary: str | None
    content_text: str | None
    authors: list[str]
    published_at: datetime | None
    source_updated_at: datetime | None
    source_categories: list[str]
    topics: list[str]
    image_url: str | None
    normalizer_version: int = NORMALIZER_VERSION

    def values(self) -> dict:
        values = asdict(self)
        # Identity and tracking parameters do not represent editorial changes.
        meaningful = {k: v for k, v in values.items() if k not in {"external_id", "original_url"}}
        serialized = json.dumps(meaningful, sort_keys=True, ensure_ascii=False, default=str)
        values["content_hash"] = hashlib.sha256(serialized.encode()).hexdigest()
        return values


@dataclass(frozen=True)
class ItemError:
    index: int
    reason: str
    item: dict


@dataclass(frozen=True)
class ParsedFeed:
    articles: list[ParsedArticle]
    errors: list[ItemError]


def map_entry(entry: dict, config: FeedConfig) -> ParsedArticle:
    title = plain_text(entry.get("title"))
    if not title:
        raise ValueError("Missing title")
    link = entry.get("link")
    if not link:
        raise ValueError("Missing article URL")
    url = canonical_url(link, config.site_url)
    categories = sorted({t["term"].strip() for t in entry.get("tags", []) if t.get("term")})
    topics = set(config.default_topics)
    for category in categories:
        lower = category.lower()
        topics.update(config.category_mapping.get(category, []))
        for topic, needles in TOPIC_RULES.items():
            if any(needle in lower for needle in needles):
                topics.add(topic)
    authors = sorted({a["name"].strip() for a in entry.get("authors", []) if a.get("name")})
    if not authors and entry.get("author"):
        authors = [plain_text(entry["author"]) or entry["author"]]
    image = None
    for enclosure in entry.get("enclosures", []) + entry.get("media_content", []):
        if enclosure.get("type", "").startswith("image/") or enclosure.get("medium") == "image":
            image = enclosure.get("href") or enclosure.get("url")
            break
    if image:
        try:
            image = canonical_url(image, config.site_url)
        except ValueError:
            image = None
    external_id = entry.get("id")
    # FeedParserDict aliases missing `updated` to `published`; keep the meanings distinct.
    updated = entry.get("updated") if "updated" in entry else None
    if config.adapter == "rbc":
        external_id = entry.get("rbc_news_news_id") or entry.get("rbc_news_id") or external_id
        updated = entry.get("rbc_news_newsmodifdate") or entry.get("rbc_newsmodifdate") or updated
    content = None
    if config.store_content:
        if config.adapter == "rbc":
            # RBC's content:encoded is an image caption, not an article body.
            content = plain_text(entry.get("rbc_news_full-text") or entry.get("rbc_full-text"))
        else:
            content = plain_text(" ".join(c.get("value", "") for c in entry.get("content", [])))
    return ParsedArticle(
        external_id=str(external_id) if external_id else None,
        original_url=urljoin(config.site_url, link),
        canonical_url=url,
        title=title,
        summary=plain_text(
            entry.get("rbc_news_anons") or entry.get("rbc_anons") or entry.get("summary")
        ),
        content_text=content,
        authors=authors,
        published_at=parse_date(
            entry.get("published"), entry.get("published_parsed"), config.default_timezone
        ),
        source_updated_at=parse_date(
            updated,
            entry.get("updated_parsed") if "updated_parsed" in entry else None,
            config.default_timezone,
        ),
        source_categories=categories,
        topics=sorted(topics or {"other"}),
        image_url=image,
    )


def parse_feed(body: bytes, config: FeedConfig) -> ParsedFeed:
    # Validate before the tolerant feedparser: no DTD, entities, external resources or HTML.
    root = fromstring(body, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    if root.tag not in {
        "rss",
        "{http://www.w3.org/2005/Atom}feed",
        "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF",
    }:
        raise ValueError("Response is not RSS or Atom")
    parsed = feedparser.parse(body)
    if not parsed.version:
        raise ValueError("Unsupported feed format")
    articles, errors = [], []
    for index, entry in enumerate(parsed.entries):
        try:
            articles.append(map_entry(entry, config))
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            errors.append(
                ItemError(
                    index,
                    str(exc)[:300],
                    {
                        "title": str(entry.get("title", ""))[:1000],
                        "link": str(entry.get("link", ""))[:2000],
                        "id": str(entry.get("id", ""))[:1000],
                    },
                )
            )
    return ParsedFeed(articles, errors)
