import asyncio
import contextlib
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.repositories.analytics import (
    DocumentClaim,
    LeaseLostError,
    claim_document,
    embedding_inputs,
    finish,
    load_original,
    renew_lease,
    requeue_unembedded,
    save_parsed,
)
from app.services.analytics.parser import ParsedDocument, PdfValidationError
from app.services.llm.embeddings import get_embedder

logger = logging.getLogger(__name__)
BACKEND_ROOT = Path(__file__).resolve().parents[3]
# Как часто пробовать повторно векторизовать документы, у которых BGE был недоступен.
REEMBED_INTERVAL_SECONDS = 300


async def extract_pdf(content: bytes, settings: Settings) -> ParsedDocument:
    """Isolate MuPDF and bound OCR runtime; cancellation terminates the parser process."""
    with TemporaryDirectory(prefix="alfa-analytics-") as folder:
        source, target = Path(folder) / "source.pdf", Path(folder) / "result.json"
        source.write_bytes(content)
        args = [
            sys.executable,
            "-m",
            "app.services.analytics.parser",
            str(source),
            str(target),
            "--languages",
            settings.analytics_ocr_languages,
            "--dpi",
            str(settings.analytics_ocr_dpi),
            "--max-pages",
            str(settings.analytics_max_pages),
        ]
        if settings.analytics_tessdata:
            args += ["--tessdata", settings.analytics_tessdata]
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=BACKEND_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
        )
        try:
            async with asyncio.timeout(settings.analytics_parse_timeout_seconds):
                await process.wait()
            if process.returncode != 0:
                if target.exists():
                    import json

                    message = json.loads(target.read_text(encoding="utf-8")).get("error")
                    raise PdfValidationError(message or "Не удалось распознать PDF")
                raise PdfValidationError("Процесс распознавания PDF завершился с ошибкой")
            result = ParsedDocument.model_validate_json(target.read_text(encoding="utf-8"))
            if not result.chunks:
                raise PdfValidationError(
                    "PDF не содержит доступных для поиска текстовых фрагментов"
                )
            return result
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()


class AnalyticsWorker:
    def __init__(self, factory: async_sessionmaker, settings: Settings):
        self.factory = factory
        self.settings = settings
        self.task: asyncio.Task | None = None

    def start(self):
        self.task = asyncio.create_task(self.run(), name="analytics-worker")

    async def stop(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    async def heartbeat(self, claim: DocumentClaim):
        while True:
            await asyncio.sleep(30)
            await renew_lease(self.factory, claim)

    async def process(self, claim: DocumentClaim):
        if claim.needs_parse:
            content = await load_original(self.factory, claim)
            parsed = await extract_pdf(content, self.settings)
            await save_parsed(self.factory, claim, parsed)
        if not self.settings.analytics_embeddings_enabled:
            await finish(
                self.factory,
                claim,
                embedding_error="Векторизация выключена; доступен полнотекстовый поиск",
            )
            return
        inputs = await embedding_inputs(self.factory, claim)
        try:
            embedder = get_embedder()
            vectors = await asyncio.to_thread(embedder.embed, [text for _, text in inputs])
            await finish(
                self.factory,
                claim,
                vectors=list(zip([key for key, _ in inputs], vectors, strict=True)),
                model_version=embedder.model_version,
            )
        except LeaseLostError:
            raise
        except Exception:
            logger.exception("Analytics embedding failed for %s", claim.document_id)
            await finish(
                self.factory,
                claim,
                embedding_error="BGE недоступен или вернул ошибку; доступен полнотекстовый поиск. "
                "После восстановления вызовите /retry",
            )

    async def run(self):
        next_reembed = time.monotonic() + 30
        while True:
            claim = None
            try:
                claim = await claim_document(self.factory)
                if claim is None:
                    if (
                        self.settings.analytics_embeddings_enabled
                        and time.monotonic() >= next_reembed
                    ):
                        next_reembed = time.monotonic() + REEMBED_INTERVAL_SECONDS
                        requeued = await requeue_unembedded(self.factory)
                        if requeued:
                            logger.info("Analytics: повторная векторизация %d документов", requeued)
                    await asyncio.sleep(2)
                    continue
                async with asyncio.TaskGroup() as group:
                    heartbeat = group.create_task(self.heartbeat(claim))
                    try:
                        await self.process(claim)
                    finally:
                        heartbeat.cancel()
            except asyncio.CancelledError:
                # Persisted leases expire after shutdown; another worker can resume the job.
                raise
            except Exception as exc:
                logger.exception("Analytics processing failed")
                if claim:
                    cause = exc
                    if isinstance(exc, ExceptionGroup):
                        cause = exc.exceptions[0]
                    message = (
                        str(cause)
                        if isinstance(cause, PdfValidationError)
                        else ("Ошибка обработки PDF; проверьте настройки и повторите /retry")
                    )
                    with contextlib.suppress(Exception):
                        await finish(self.factory, claim, error=message)
                await asyncio.sleep(2)
