"""Run against a dedicated migrated database via ANALYTICS_TEST_POSTGRES_DB."""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.dependencies import get_current_user
from app.core.config import Settings
from app.main import create_app
from app.models.analytics import AnalyticsChunk, AnalyticsDocument
from app.models.user import User
from app.repositories.analytics import (
    LeaseLostError,
    claim_document,
    enqueue,
    finish,
    renew_lease,
    requeue_unembedded,
)
from app.schemas.analytics import SearchRequest
from app.services.analytics.search import search_analytics
from app.services.analytics.worker import AnalyticsWorker
from tests.test_analytics import sample_pdf


@pytest.fixture
def settings():
    name = os.environ.get("ANALYTICS_TEST_POSTGRES_DB")
    if not name:
        pytest.skip("Set ANALYTICS_TEST_POSTGRES_DB to a dedicated migrated PostgreSQL database")
    return Settings(
        postgres_db=name,
        rss_enabled=False,
        analytics_enabled=False,
        analytics_embeddings_enabled=False,
    )


@asynccontextmanager
async def sessions(settings):
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


def test_upload_parse_search_retry_and_original(settings, monkeypatch):
    user_id = uuid4()
    body = sample_pdf()

    async def setup():
        async with sessions(settings) as factory, factory.begin() as session:
            session.add(
                User(user_id=user_id, email=f"{user_id}@test.invalid", password_hash="unused")
            )

    async def process():
        async with sessions(settings) as factory:
            claim = await claim_document(factory)
            assert claim is not None
            assert await claim_document(factory) is None
            await AnalyticsWorker(factory, settings).process(claim)

    async def cleanup():
        async with sessions(settings) as factory, factory.begin() as session:
            await session.execute(
                delete(AnalyticsDocument).where(
                    AnalyticsDocument.uploaded_by == user_id,
                )
            )
            await session.execute(delete(User).where(User.user_id == user_id))

    asyncio.run(setup())
    app = create_app(settings)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=user_id)
    try:
        with TestClient(app) as client:
            fields = {"publisher": "Test research", "topics": "inflation,rates"}
            response = client.post(
                "/analytics/documents",
                data=fields,
                files={"file": ("report.pdf", body)},
            )
            assert response.status_code == 202, response.text
            document = response.json()["document"]
            identifier = document["id"]
            assert document["status"] == "queued"
            duplicate = client.post(
                "/analytics/documents",
                data=fields,
                files={"file": ("copy.pdf", body)},
            )
            assert duplicate.json()["duplicate"] is True
            assert duplicate.json()["document"]["id"] == identifier
            assert client.post(f"/analytics/documents/{identifier}/retry").status_code == 409
            asyncio.run(process())
            document = client.get(f"/analytics/documents/{identifier}").json()
            assert document["status"] == "parsed"
            assert document["published_at"] == "2026-09-11"
            assert document["chunk_count"] > 0
            assert document["embedding_error"]
            amended = client.patch(
                f"/analytics/documents/{identifier}",
                json={
                    "published_at": "2026-09-11",
                    "topics": ["inflation"],
                },
            )
            assert amended.status_code == 200
            assert amended.json()["date_origin"] == "provided"
            assert client.get(f"/analytics/documents/{identifier}/file").content == body
            passages = client.get(f"/analytics/documents/{identifier}/passages").json()
            assert passages[0]["page_number"] == 1
            page = client.get(f"/analytics/documents/{identifier}/pages/1").json()
            assert page["page"]["method"] == "native"
            result = client.post(
                "/analytics/search", json={"query": "inflation", "mode": "keyword"}
            )
            assert result.status_code == 200, result.text
            hit = result.json()["items"][0]
            assert hit["document_id"] == identifier
            assert hit["publisher"] == fields["publisher"]
            assert hit["source_url"].endswith("#page=1")
            assert hit["evidence_ref"].startswith("analytics:")
            earlier = client.post(
                "/analytics/search",
                json={
                    "query": "inflation",
                    "mode": "keyword",
                    "as_of": "2026-09-10",
                },
            )
            assert earlier.json()["items"] == []
            assert client.post(f"/analytics/documents/{identifier}/retry").status_code == 202
            # Re-indexing a parsed document must not repeat PDF extraction.
            monkeypatch.setattr(
                "app.services.analytics.worker.extract_pdf",
                lambda *_: pytest.fail("repeated OCR"),
            )

            class FakeEmbedder:
                model_version = "test-bge:3d"

                def embed(self, texts):
                    return [[1.0, 0.0, 0.0] for _ in texts]

                def embed_one(self, text):
                    return [1.0, 0.0, 0.0]

            embedder = FakeEmbedder()
            monkeypatch.setattr("app.services.analytics.worker.get_embedder", lambda: embedder)
            monkeypatch.setattr("app.services.analytics.search.get_embedder", lambda: embedder)
            settings.analytics_embeddings_enabled = True
            asyncio.run(process())
            assert client.get(f"/analytics/documents/{identifier}").json()["status"] == "ready"
            result = client.post("/analytics/search", json={"query": "inflation"}).json()
            assert result["mode_used"] == "hybrid"
            assert result["items"][0]["document_id"] == identifier

            def broken_embed(text):
                raise RuntimeError("offline")

            monkeypatch.setattr(embedder, "embed_one", broken_embed)
            fallback = client.post("/analytics/search", json={"query": "inflation"}).json()
            assert fallback["mode_used"] == "keyword"
            assert fallback["items"]
            assert (
                client.post(
                    "/analytics/search",
                    json={
                        "query": "inflation",
                        "mode": "semantic",
                    },
                ).status_code
                == 503
            )
            monkeypatch.setattr(embedder, "embed", broken_embed)
            assert client.post(f"/analytics/documents/{identifier}/retry").status_code == 202
            asyncio.run(process())
            failed_embedding = client.get(f"/analytics/documents/{identifier}").json()
            assert failed_embedding["status"] == "parsed"
            assert failed_embedding["embedding_error"]
            assert client.get(f"/analytics/documents/{identifier}/passages").json() == passages
            updated = client.patch(
                f"/analytics/documents/{identifier}",
                json={
                    "title": "Updated research title",
                },
            )
            assert updated.json()["status"] == "indexing"
            assert updated.json()["embedding_model_version"] is None
    finally:
        asyncio.run(cleanup())


def test_expired_lease_cannot_publish_and_undated_sources_are_filtered(settings):
    async def scenario():
        async with sessions(settings) as factory:
            user_id = uuid4()
            async with factory.begin() as session:
                session.add(
                    User(
                        user_id=user_id,
                        email=f"{user_id}@test.invalid",
                        password_hash="unused",
                    )
                )
            try:
                async with factory() as session:
                    record, _ = await enqueue(
                        session,
                        sha256=uuid4().hex,
                        filename="source.pdf",
                        title="Research",
                        publisher="Test",
                        uploaded_by=user_id,
                        content=b"%PDF-",
                        topics=[],
                    )
                old_claim = await claim_document(factory)
                async with factory.begin() as session:
                    await session.execute(
                        update(AnalyticsDocument)
                        .where(
                            AnalyticsDocument.id == record.id,
                        )
                        .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
                    )
                new_claim = await claim_document(factory)
                assert new_claim.token != old_claim.token
                with pytest.raises(LeaseLostError):
                    await renew_lease(factory, old_claim)
                with pytest.raises(LeaseLostError):
                    await finish(factory, old_claim, error="stale worker")
                await finish(factory, new_claim)
                async with factory.begin() as session:
                    session.add(
                        AnalyticsChunk(
                            document_id=record.id,
                            chunk_id="p1-c1",
                            page_number=1,
                            section="Research",
                            text="inflation",
                            search_text="inflation",
                            bbox=[0, 0, 1, 1],
                        )
                    )
                async with factory() as session:
                    all_hits = await search_analytics(
                        session,
                        SearchRequest(
                            query="inflation",
                            mode="keyword",
                        ),
                    )
                    assert all_hits.items
                    dated = await search_analytics(
                        session,
                        SearchRequest(
                            query="inflation",
                            mode="keyword",
                            as_of="2026-10-23",
                        ),
                    )
                    assert dated.items == []
                    assert (
                        await session.scalar(
                            select(AnalyticsDocument.status).where(
                                AnalyticsDocument.id == record.id,
                            )
                        )
                        == "parsed"
                    )
            finally:
                async with factory.begin() as session:
                    await session.execute(
                        delete(AnalyticsDocument).where(
                            AnalyticsDocument.uploaded_by == user_id,
                        )
                    )
                    await session.execute(delete(User).where(User.user_id == user_id))

    asyncio.run(scenario())


def test_documents_without_vectors_are_requeued_for_embedding(settings):
    user_id = uuid4()

    async def scenario():
        async with sessions(settings) as factory:
            async with factory.begin() as session:
                session.add(
                    User(user_id=user_id, email=f"{user_id}@test.invalid", password_hash="unused")
                )
            try:
                ids = {}
                for name, status, chunks in (
                    ("lost", "parsed", 3),
                    ("ready", "ready", 3),
                    ("empty", "parsed", 0),
                ):
                    async with factory() as session:
                        record, _ = await enqueue(
                            session,
                            sha256=uuid4().hex,
                            filename=f"{name}.pdf",
                            title=name,
                            publisher="Test",
                            uploaded_by=user_id,
                            content=b"%PDF-",
                            topics=[],
                        )
                    ids[name] = record.id
                    async with factory.begin() as session:
                        await session.execute(
                            update(AnalyticsDocument)
                            .where(AnalyticsDocument.id == record.id)
                            .values(
                                status=status,
                                chunk_count=chunks,
                                embedding_error="BGE недоступен",
                            )
                        )
                requeued = await requeue_unembedded(factory)
                async with factory() as session:
                    rows = {
                        row.id: row
                        for row in await session.scalars(
                            select(AnalyticsDocument).where(AnalyticsDocument.id.in_(ids.values()))
                        )
                    }
                return requeued, ids, rows
            finally:
                async with factory.begin() as session:
                    await session.execute(
                        delete(AnalyticsDocument).where(AnalyticsDocument.uploaded_by == user_id)
                    )
                    await session.execute(delete(User).where(User.user_id == user_id))

    requeued, ids, rows = asyncio.run(scenario())
    assert requeued >= 1
    assert rows[ids["lost"]].status == "indexing"
    assert rows[ids["lost"]].embedding_error is None
    assert rows[ids["ready"]].status == "ready"
    assert rows[ids["empty"]].status == "parsed"


def test_keyword_only_hits_get_semantic_score_for_similarity_threshold(settings, monkeypatch):
    user_id = uuid4()

    class FakeEmbedder:
        model_version = "fake:4d"

        def embed_one(self, text):
            return [1.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr("app.services.analytics.search.get_embedder", lambda: FakeEmbedder())

    async def scenario():
        async with sessions(settings) as factory:
            async with factory.begin() as session:
                session.add(
                    User(user_id=user_id, email=f"{user_id}@test.invalid", password_hash="unused")
                )
            try:
                async with factory() as session:
                    record, _ = await enqueue(
                        session,
                        sha256=uuid4().hex,
                        filename="forecast.pdf",
                        title="Прогноз",
                        publisher="Test",
                        uploaded_by=user_id,
                        content=b"%PDF-",
                        topics=[],
                    )
                async with factory.begin() as session:
                    await session.execute(
                        update(AnalyticsDocument)
                        .where(AnalyticsDocument.id == record.id)
                        .values(status="ready", chunk_count=2, published_at=datetime(2026, 10, 1))
                    )
                    for chunk_id, vector in (("near", [1.0, 0, 0, 0]), ("far", [0, 1.0, 0, 0])):
                        text = f"Ключевая ставка и прогноз {chunk_id}"
                        session.add(
                            AnalyticsChunk(
                                document_id=record.id,
                                chunk_id=chunk_id,
                                page_number=1,
                                section="Раздел",
                                text=text,
                                bbox=[0, 0, 1, 1],
                                search_text=text,
                                vector=vector,
                                embedding_model_version="fake:4d",
                            )
                        )
                async with factory() as session:
                    return await search_analytics(
                        session,
                        SearchRequest(
                            query="ставка", semantic_query="ставка", document_ids=[record.id]
                        ),
                    )
            finally:
                async with factory.begin() as session:
                    await session.execute(
                        delete(AnalyticsDocument).where(AnalyticsDocument.uploaded_by == user_id)
                    )
                    await session.execute(delete(User).where(User.user_id == user_id))

    response = asyncio.run(scenario())
    scores = {hit.chunk_id: hit.semantic_score for hit in response.items}
    assert scores["near"] == 1.0
    # «far» найден по слову, но смыслово ортогонален: оценка 0.0, а не None
    assert scores["far"] == 0.0
