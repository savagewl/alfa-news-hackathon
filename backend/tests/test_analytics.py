import asyncio
from datetime import date
from uuid import uuid4

import pymupdf
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app
from app.schemas.analytics import SearchRequest
from app.services.analytics.parser import (
    ParsedPage,
    PdfValidationError,
    TextBlock,
    chunk_page,
    detect_publication_date,
    inspect_pdf,
    parse_pdf,
    publication_date,
)
from app.services.analytics.worker import extract_pdf


def sample_pdf(text: str | None = None) -> bytes:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            (40, 40, 550, 500),
            text
            or (
                f"11.09.2026\nInflation and policy rate outlook. {uuid4()}\n"
                "We expect the policy rate to remain unchanged through December. "
                "This is a forecast rather than a confirmed future observation."
            ),
            fontsize=12,
        )
        return pdf.tobytes()


def test_native_pdf_preserves_source_and_geometry():
    parsed = parse_pdf(sample_pdf())
    assert parsed.detected_date == date(2026, 9, 11)
    assert parsed.pages[0].method == "native"
    assert parsed.chunks[0].page_number == 1
    assert "Inflation" in parsed.chunks[0].text
    assert all(len(c.bbox) == 4 and c.bbox[2] > c.bbox[0] for c in parsed.chunks)
    assert len(parsed.sha256) == 64


def test_extraction_does_not_depend_on_server_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parsed = asyncio.run(extract_pdf(sample_pdf(), Settings(_env_file=None)))
    assert parsed.detected_date == date(2026, 9, 11)
    assert parsed.chunks


def test_small_numeric_cells_are_preserved():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 40), "Research about inflation and policy rates in the coming year")
        page.insert_text((40, 200), "3.5%")
        parsed = parse_pdf(pdf.tobytes())
    assert "3.5%" in "\n".join(c.text for c in parsed.chunks)


def test_invalid_protected_and_oversize_page_counts():
    with pytest.raises(PdfValidationError):
        inspect_pdf(b"not a pdf")
    with pytest.raises(PdfValidationError):
        inspect_pdf(b"%PDF-invalid")
    with pymupdf.open(stream=sample_pdf(), filetype="pdf") as pdf:
        protected = pdf.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="password",
        )
        pdf.new_page()
        two_pages = pdf.tobytes()
    with pytest.raises(PdfValidationError, match="паролем"):
        inspect_pdf(protected)
    with pytest.raises(PdfValidationError, match="страниц"):
        inspect_pdf(two_pages, max_pages=1)


def test_chunking_keeps_content_and_excludes_service_blocks():
    text = " ".join(f"word{i}" for i in range(1800))
    page = ParsedPage(
        number=7,
        width=600,
        height=800,
        method="ocr",
        text=text,
        blocks=[
            TextBlock(bbox=(10, 10, 500, 600), text=text),
            TextBlock(bbox=(10, 750, 500, 790), text="legal footer", excluded=True),
        ],
    )
    chunks = chunk_page(page)
    assert len(chunks) > 2
    assert all(len(c.text) <= 1800 and c.page_number == 7 for c in chunks)
    assert " ".join(c.text for c in chunks) == text
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunk_page(page.model_copy(update={"excluded": True})) == []


def test_dates_require_an_explicit_valid_day_month_year():
    assert publication_date("15 апреля 2021 года") == date(2021, 4, 15)
    assert publication_date("22 июля 2026 г.") == date(2026, 7, 22)
    assert publication_date("Август 2026") is None
    assert publication_date("31 февраля 2026") is None
    assert publication_date("31.02.2026") is None


def test_copyright_footer_does_not_supply_publication_date():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text(
            (40, 40), "A sufficiently long financial research headline about inflation"
        )
        page.insert_text((40, 800), "Copyright 2026. License dated 16.01.2015")
        parsed = parse_pdf(pdf.tobytes())
    assert parsed.detected_date is None


def test_bond_maturity_date_is_not_used_as_publication_date():
    blocks = [
        TextBlock(bbox=(30, 200, 500, 400), text="Погашение 17 сентября 2031 года"),
        TextBlock(bbox=(30, 780, 200, 800), text="14 сентября 2026"),
    ]
    assert detect_publication_date(blocks, 840) == date(2026, 9, 14)
    assert detect_publication_date(blocks[:1], 840) is None


def test_search_rejects_empty_query_and_reversed_period():
    with pytest.raises(ValidationError):
        SearchRequest(query="  ")
    with pytest.raises(ValidationError):
        SearchRequest(query="inflation", published_from="2026-09-01", as_of="2026-08-01")


def test_analytics_api_is_authenticated_and_documented():
    with TestClient(create_app(Settings(_env_file=None))) as client:
        assert client.get("/analytics/documents").status_code == 401
        assert client.post("/analytics/search", json={"query": "inflation"}).status_code == 401
        schema = client.get("/openapi.json").json()
        upload = schema["paths"]["/analytics/documents"]["post"]
        assert "multipart/form-data" in upload["requestBody"]["content"]
        assert "202" in upload["responses"]
        assert "/analytics/documents/{document_id}/pages/{number}" in schema["paths"]
