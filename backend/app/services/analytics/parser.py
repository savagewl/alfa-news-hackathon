"""Read-only PDF extraction. Run in a separate process: MuPDF is not thread-safe."""

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pymupdf
from pydantic import BaseModel, Field

PARSER_VERSION = "pdf-v1"


class PdfValidationError(ValueError):
    pass


class TextBlock(BaseModel):
    bbox: tuple[float, float, float, float]
    text: str
    excluded: bool = False


class ParsedPage(BaseModel):
    number: int
    width: float
    height: float
    method: str
    text: str
    blocks: list[TextBlock]
    excluded: bool = False


class ParsedChunk(BaseModel):
    chunk_id: str
    page_number: int
    section: str
    text: str
    bbox: tuple[float, float, float, float]


class ParsedDocument(BaseModel):
    sha256: str
    parser_version: str = PARSER_VERSION
    detected_date: date | None = None
    pages: list[ParsedPage]
    chunks: list[ParsedChunk]
    warnings: list[str] = Field(default_factory=list)


_MONTHS = {
    name: i
    for i, name in enumerate(
        (
            "января",
            "февраля",
            "марта",
            "апреля",
            "мая",
            "июня",
            "июля",
            "августа",
            "сентября",
            "октября",
            "ноября",
            "декабря",
        ),
        1,
    )
}
_SERVICE = re.compile(
    r"контакты\s*(?:и|/)\s*дискле[йи]мер|дискле[йи]мер|^контакты\s*$|"
    r"все права защищены.*?генеральная лицензия",
    re.I | re.S,
)


def publication_date(text: str) -> date | None:
    """Only an explicit date; never infer a year from a filename or copyright footer."""
    match = re.search(r"\b(\d{1,2})\s+(" + "|".join(_MONTHS) + r")\s+(20\d{2})\b", text, re.I)
    if match:
        day, month, year = match.groups()
        parts = int(year), _MONTHS[month.lower()], int(day)
    else:
        match = re.search(r"\b(\d{2})[./](\d{2})[./](20\d{2})\b", text)
        if not match:
            return None
        day, month, year = match.groups()
        parts = int(year), int(month), int(day)
    try:
        return date(*parts)
    except ValueError:
        return None


def detect_publication_date(blocks: list[TextBlock], page_height: float) -> date | None:
    # Dates in running text may be bond maturities or historic events.
    pattern = re.compile(
        r"(?:\d{1,2}\s+(?:" + "|".join(_MONTHS) + r")\s+20\d{2}|"
        r"\d{2}[./]\d{2}[./]20\d{2})\s*(?:[гr]\.?|года)?[.,]?",
        re.I,
    )
    for block in blocks:
        if block.excluded or not (
            block.bbox[1] < page_height * 0.35 or block.bbox[1] > page_height * 0.75
        ):
            continue
        for line in block.text.splitlines():
            if pattern.fullmatch(line.strip()):
                result = publication_date(line)
                if result:
                    return result
    return None


def _clean(text: str) -> str:
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = re.sub(r"(?<=[а-яa-z])-\n(?=[а-яa-z])", "", text)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()).strip()


def exclude_block(text: str, bbox: tuple | list, page_height: float) -> bool:
    footer = bbox[1] > page_height * 0.88
    legal = bool(re.search(r"генеральная лицензия|все права защищены", text, re.I))
    # Small numeric cells are evidence too; only page numbers at the bottom are decorative.
    decorative = (len(text) <= 2 and not any(c.isdigit() for c in text)) or (
        text.isdigit() and len(text) < 5 and bbox[1] > page_height * 0.95
    )
    return (footer and legal) or decorative


def _union(blocks: list[TextBlock]) -> tuple[float, float, float, float]:
    return (
        min(b.bbox[0] for b in blocks),
        min(b.bbox[1] for b in blocks),
        max(b.bbox[2] for b in blocks),
        max(b.bbox[3] for b in blocks),
    )


def chunk_page(page: ParsedPage, max_chars: int = 1800) -> list[ParsedChunk]:
    """Keep page provenance; split oversized blocks without dropping any characters."""
    if page.excluded:
        return []
    result = []
    group: list[TextBlock] = []
    section = ""

    def emit():
        if group:
            result.append(
                ParsedChunk(
                    chunk_id=f"p{page.number}-c{len(result) + 1}",
                    page_number=page.number,
                    section=section,
                    text="\n\n".join(b.text for b in group),
                    bbox=_union(group),
                )
            )
            group.clear()

    for block in page.blocks:
        if block.excluded or not block.text:
            continue
        if not section:
            section = next((line[:180] for line in block.text.splitlines() if len(line) >= 15), "")
        remaining = block.text
        while remaining:
            if len(remaining) > max_chars:
                split = remaining.rfind(" ", 0, max_chars)
                split = split if split > max_chars // 2 else max_chars
            else:
                split = len(remaining)
            piece = remaining[:split].strip()
            remaining = remaining[split:].strip()
            if group and sum(len(b.text) + 2 for b in group) + len(piece) > max_chars:
                emit()
            if piece:
                group.append(block.model_copy(update={"text": piece}))
    emit()
    return result


def inspect_pdf(content: bytes, max_pages: int = 100) -> int:
    if not content.startswith(b"%PDF-"):
        raise PdfValidationError("Файл должен быть PDF")
    try:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise PdfValidationError("PDF защищён паролем")
            if not 0 < len(pdf) <= max_pages:
                raise PdfValidationError(f"Допустимо от 1 до {max_pages} страниц")
            return len(pdf)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as exc:
        raise PdfValidationError("Повреждённый или неподдерживаемый PDF") from exc


def parse_pdf(
    content: bytes,
    *,
    tessdata: str | None = None,
    languages: str = "rus+eng",
    dpi: int = 200,
    max_pages: int = 100,
) -> ParsedDocument:
    inspect_pdf(content, max_pages)
    pages = []
    chunks = []
    warnings = []
    detected_date = None
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        for number, page in enumerate(pdf, 1):
            if page.rect.is_empty or page.rect.width * page.rect.height * (dpi / 72) ** 2 > 20e6:
                raise PdfValidationError(f"Страница {number}: слишком большой размер изображения")
            native = page.get_text("text")
            method = "native"
            textpage = None
            if len(re.sub(r"\W", "", native)) < 40:
                method = "ocr"
                try:
                    textpage = page.get_textpage_ocr(
                        language=languages,
                        dpi=dpi,
                        full=True,
                        tessdata=tessdata,
                    )
                except RuntimeError as exc:
                    raise PdfValidationError(
                        "OCR недоступен: установите языки rus и eng и укажите ANALYTICS_TESSDATA"
                    ) from exc
            # MuPDF/Tesseract block order retains columns better than global y/x sorting.
            raw = page.get_text("blocks", textpage=textpage, sort=False)
            blocks = []
            for item in raw:
                if item[6] != 0:
                    continue
                text = _clean(item[4])
                if text:
                    blocks.append(
                        TextBlock(
                            bbox=item[:4],
                            text=text,
                            excluded=exclude_block(text, item[:4], page.rect.height),
                        )
                    )
            full_text = "\n\n".join(b.text for b in blocks)
            excluded = bool(_SERVICE.search(full_text[:450]))
            record = ParsedPage(
                number=number,
                width=page.rect.width,
                height=page.rect.height,
                method=method,
                text=full_text,
                blocks=blocks,
                excluded=excluded,
            )
            pages.append(record)
            chunks.extend(chunk_page(record))
            if number == 1:
                detected_date = detect_publication_date(blocks, page.rect.height)
            if not excluded and len(re.sub(r"\W", "", full_text)) < 10:
                warnings.append(f"Страница {number}: текст не распознан или почти отсутствует")
    if any(p.method == "ocr" for p in pages):
        warnings.append("Использован OCR: числа, таблицы и порядок чтения требуют проверки")
    warnings.append("Таблицы и графики сохранены как текстовые блоки; числовые ряды не проверены")
    if not detected_date:
        warnings.append("Дата публикации не определена; задайте её явно")
    return ParsedDocument(
        sha256=hashlib.sha256(content).hexdigest(),
        detected_date=detected_date,
        pages=pages,
        chunks=chunks,
        warnings=warnings,
    )


def main():
    cli = argparse.ArgumentParser(description="Extract PDF pages and citation-ready chunks as JSON")
    cli.add_argument("input", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--tessdata")
    cli.add_argument("--languages", default="rus+eng")
    cli.add_argument("--dpi", type=int, default=200)
    cli.add_argument("--max-pages", type=int, default=100)
    args = cli.parse_args()
    try:
        result = parse_pdf(
            args.input.read_bytes(),
            tessdata=args.tessdata,
            languages=args.languages,
            dpi=args.dpi,
            max_pages=args.max_pages,
        )
        args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    except PdfValidationError as exc:
        args.output.write_text(
            json.dumps({"error": str(exc)}, ensure_ascii=False), encoding="utf-8"
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
