"""Restore trustworthy RSS bodies before starting article page fetching.

Run after migrations: python -m app.services.rss.backfill
Existing HTML bodies and newer editorial revisions are preserved.
"""

import asyncio
import zlib

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import Database
from app.models.news import NewsArticle, RssSnapshot
from app.repositories.article_content import set_article_content
from app.services.rss.catalog import load_catalog
from app.services.rss.parser import parse_feed


async def backfill_rss(factory) -> int:
    configs = {c.feed_code: c for c in load_catalog() if c.store_content}
    async with factory() as session:
        ids = (await session.scalars(
            select(RssSnapshot.id).where(RssSnapshot.feed_code.in_(configs))
            .order_by(RssSnapshot.created_at.desc(), RssSnapshot.id)
        )).all()
    seen = set()
    restored = 0
    for snapshot_id in ids:
        async with factory() as session:
            snapshot = await session.get(RssSnapshot, snapshot_id)
            if snapshot is None:
                continue
            config = configs[snapshot.feed_code]
            body = snapshot.compressed_body
        parsed = await asyncio.to_thread(parse_feed, zlib.decompress(body), config)
        for incoming in parsed.articles:
            key = (config.code, incoming.canonical_url)
            if key in seen or not incoming.content_text:
                continue
            seen.add(key)
            async with factory.begin() as session:
                article = await session.scalar(
                    select(NewsArticle)
                    .where(NewsArticle.source_code == config.code,
                           NewsArticle.canonical_url == incoming.canonical_url,
                           NewsArticle.content_origin.is_(None))
                    .with_for_update()
                )
                if article is None:
                    continue
                if article.source_updated_at and (
                    incoming.source_updated_at is None
                    or incoming.source_updated_at < article.source_updated_at
                ):
                    continue
                set_article_content(article, incoming.content_text, "rss")
                restored += 1
    return restored


async def main() -> None:
    database = Database(get_settings())
    try:
        print(f"Restored RSS article bodies: {await backfill_rss(database.session_factory)}")
    finally:
        await database.close()


if __name__ == "__main__":
    asyncio.run(main())
