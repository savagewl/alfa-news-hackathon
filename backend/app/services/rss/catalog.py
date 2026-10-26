import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeedConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    name: str
    site_url: str
    feed_code: str
    url: str
    interval: int = Field(ge=30)
    allowed_hosts: list[str]
    adapter: str = "generic"
    enabled: bool = True
    kind: str = "media"
    default_timezone: str = "Europe/Moscow"
    guid_scope: str = "feed"
    store_content: bool = False
    default_topics: list[str] = Field(default_factory=list)
    category_mapping: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_config(self):
        from urllib.parse import urlsplit
        from zoneinfo import ZoneInfo

        if self.adapter not in {"generic", "rbc"}:
            raise ValueError("Unknown RSS adapter")
        if self.guid_scope not in {"feed", "source"}:
            raise ValueError("GUID scope must be feed or source")
        if urlsplit(self.url).scheme != "https":
            raise ValueError("RSS feeds require HTTPS")
        if urlsplit(self.url).hostname not in self.allowed_hosts:
            raise ValueError("RSS URL host is not allowed")
        ZoneInfo(self.default_timezone)
        return self


@lru_cache
def load_catalog() -> tuple[FeedConfig, ...]:
    content = Path(__file__).with_name("sources.json").read_text(encoding="utf-8")
    feeds = tuple(FeedConfig.model_validate(item) for item in json.loads(content))
    if len({f.feed_code for f in feeds}) != len(feeds):
        raise ValueError("Duplicate feed codes")
    return feeds
