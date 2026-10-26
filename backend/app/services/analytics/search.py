"""Hybrid retrieval for a small corpus, without requiring a PostgreSQL extension."""

import asyncio
import heapq

from sqlalchemy import func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analytics import AnalyticsChunk, AnalyticsDocument
from app.schemas.analytics import SearchHit, SearchRequest, SearchResponse
from app.services.llm.embeddings import cosine, get_embedder


class SemanticSearchUnavailable(RuntimeError):
    pass


def _filters(request: SearchRequest):
    conditions = [AnalyticsDocument.status.in_(["parsed", "ready"])]
    if request.publisher:
        conditions.append(AnalyticsDocument.publisher == request.publisher)
    if request.topic:
        conditions.append(AnalyticsDocument.topics.contains([request.topic]))
    if request.document_ids:
        conditions.append(AnalyticsDocument.id.in_(request.document_ids))
    if request.published_from:
        conditions.append(AnalyticsDocument.published_at >= request.published_from)
    if request.as_of:
        # Unknown dates must not leak future knowledge into a historical interpretation.
        conditions.append(AnalyticsDocument.published_at <= request.as_of)
    return conditions


async def search_analytics(session: AsyncSession, request: SearchRequest) -> SearchResponse:
    filters = _filters(request)
    keyword_keys = []
    semantic_keys = []
    semantic_scores: dict = {}
    query_vector = None
    compatible_version = None
    warnings = [
        "Материалы содержат непроверенный текст PDF/OCR. "
        "Проверяйте числовые утверждения по оригиналу.",
        "Дата публикации ограничивает доступность источника, "
        "но не доказывает актуальность прогноза.",
    ]
    if request.mode != "semantic":
        language = literal_column("'russian'")
        vector = func.to_tsvector(language, AnalyticsChunk.search_text)
        query = func.websearch_to_tsquery(language, request.query)
        rows = await session.execute(
            select(AnalyticsChunk.document_id, AnalyticsChunk.chunk_id)
            .join(AnalyticsDocument)
            .where(*filters, vector.op("@@")(query))
            .order_by(
                func.ts_rank_cd(vector, query).desc(),
                AnalyticsChunk.document_id,
                AnalyticsChunk.chunk_id,
            )
            .limit(60)
        )
        keyword_keys = list(rows.tuples())
    if request.mode != "keyword":
        embedder = get_embedder()
        compatible = filters + [
            AnalyticsChunk.vector.is_not(None),
            AnalyticsChunk.embedding_model_version == embedder.model_version,
        ]
        exists = await session.scalar(
            select(AnalyticsChunk.chunk_id).join(AnalyticsDocument).where(*compatible).limit(1)
        )
        if exists is None:
            if request.mode == "semantic":
                raise SemanticSearchUnavailable("Нет фрагментов с совместимыми векторами BGE")
            warnings.append("Векторов для выбранных источников нет; использован текстовый поиск")
        else:
            try:
                compatible_version = embedder.model_version
                query_vector = await asyncio.to_thread(
                    embedder.embed_one, request.semantic_query or request.query
                )
            except Exception as exc:
                if request.mode == "semantic":
                    raise SemanticSearchUnavailable("BGE недоступен") from exc
                warnings.append("BGE недоступен; использован текстовый поиск")
            else:
                heap = []
                # Exact scan is deliberate for the initial small corpus; bounded memory.
                rows = await session.stream(
                    select(
                        AnalyticsChunk.document_id, AnalyticsChunk.chunk_id, AnalyticsChunk.vector
                    )
                    .join(AnalyticsDocument)
                    .where(*compatible)
                    .execution_options(yield_per=100)
                )
                async for document_id, chunk_id, vector in rows:
                    score = cosine(query_vector, vector)
                    if score <= 0:
                        continue
                    candidate = (score, document_id, chunk_id)
                    if len(heap) < 60:
                        heapq.heappush(heap, candidate)
                    else:
                        heapq.heappushpop(heap, candidate)
                semantic_keys = [(doc, chunk) for _, doc, chunk in sorted(heap, reverse=True)]
                semantic_scores = {(doc, chunk): score for score, doc, chunk in heap}
    scores = {}
    for keys in (keyword_keys, semantic_keys):
        for rank, key in enumerate(keys, 1):
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
    selected = sorted(scores, key=lambda key: (-scores[key], str(key)))[: request.limit]
    items = []
    # At most 30 result lookups; vectors are used only to score hits found by keywords alone
    # and are never sent to the API consumer.
    for document_id, chunk_id in selected:
        chunk, document = (
            await session.execute(
                select(AnalyticsChunk, AnalyticsDocument)
                .join(AnalyticsDocument)
                .where(
                    AnalyticsChunk.document_id == document_id, AnalyticsChunk.chunk_id == chunk_id
                )
            )
        ).one()
        semantic_score = semantic_scores.get((document_id, chunk_id))
        if (
            semantic_score is None
            and query_vector is not None
            and chunk.vector
            and chunk.embedding_model_version == compatible_version
        ):
            # Найдено только по словам: без оценки смысла оно обходило бы порог сходства.
            semantic_score = cosine(query_vector, chunk.vector)
        items.append(
            SearchHit(
                document_id=document.id,
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                section=chunk.section,
                text=chunk.text,
                bbox=chunk.bbox,
                title=document.title,
                publisher=document.publisher,
                published_at=document.published_at,
                score=scores[(document_id, chunk_id)],
                semantic_score=semantic_score,
                evidence_ref=f"analytics:{document.id}#{chunk.chunk_id}",
                source_url=f"/analytics/documents/{document.id}/file#page={chunk.page_number}",
            )
        )
    mode_used = request.mode if semantic_keys or request.mode == "keyword" else "keyword"
    return SearchResponse(items=items, mode_used=mode_used, warnings=warnings)
