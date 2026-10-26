"""Offline folder extraction using the same isolated parser as the upload worker."""

import argparse
import asyncio
import json
from pathlib import Path

from app.core.config import Settings
from app.services.analytics.parser import ParsedDocument
from app.services.analytics.worker import extract_pdf


def write_corpus(manifest: list[dict], output: Path, publisher: str | None = None):
    """One passage per line, ready for subsequent ingestion or embedding."""
    with (output / "corpus.jsonl").open("w", encoding="utf-8") as stream:
        for item in manifest:
            if "error" in item:
                continue
            parsed = ParsedDocument.model_validate_json(
                (output / item["output"]).read_text(encoding="utf-8")
            )
            for chunk in parsed.chunks:
                record = {
                    "source_sha256": parsed.sha256,
                    "filename": item["filename"],
                    "publisher": publisher,
                    "published_at": item["published_at"],
                    "parser_version": parsed.parser_version,
                    "quality": "unverified",
                    "evidence_ref": f"pdf:{parsed.sha256}#{chunk.chunk_id}",
                    **chunk.model_dump(mode="json"),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


async def parse_folder(
    folder: Path,
    output: Path,
    settings: Settings,
    publisher: str | None = None,
) -> int:
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=True)
    manifest = []
    sources = await asyncio.to_thread(lambda: sorted(folder.glob("*.pdf")))
    for source in sources:
        target = output / f"{source.stem}.json"
        try:
            parsed = await extract_pdf(source.read_bytes(), settings)
            target.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
            item = {
                "filename": source.name,
                "sha256": parsed.sha256,
                "published_at": parsed.detected_date.isoformat() if parsed.detected_date else None,
                "pages": len(parsed.pages),
                "chunks": len(parsed.chunks),
                "excluded_pages": [p.number for p in parsed.pages if p.excluded],
                "output": target.name,
                "warnings": parsed.warnings,
            }
        except Exception as exc:
            item = {"filename": source.name, "error": str(exc)}
        manifest.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    await asyncio.to_thread(write_corpus, manifest, output, publisher)
    return 1 if not manifest or any("error" in item for item in manifest) else 0


def main():
    cli = argparse.ArgumentParser(
        description="Parse a folder of analytical PDFs without a database"
    )
    cli.add_argument("folder", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--tessdata")
    cli.add_argument("--publisher", help="Optional publisher attribution for the JSONL export")
    args = cli.parse_args()
    settings = Settings()
    if args.tessdata:
        settings.analytics_tessdata = args.tessdata
    if not args.folder.is_dir():
        cli.error("PDF folder does not exist")
    raise SystemExit(asyncio.run(parse_folder(args.folder, args.output, settings, args.publisher)))


if __name__ == "__main__":
    main()
