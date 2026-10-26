from hashlib import sha256
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.company import CsvImport
from app.repositories.csv_import import (
    find_import,
    lock_imports,
    save_companies,
    save_payments,
    save_persons,
)
from app.repositories.import_access import authorize_import, join_new_companies
from app.schemas.csv_import import CsvEncoding, ImportCounts, ImportKind, ImportResponse
from app.services.csv_parser import ParsedCsv


def response(record: CsvImport, *, reused: bool) -> ImportResponse:
    skipped = record.counts.get("skipped_duplicate_rows", 0)
    warnings = []
    conflicts = record.counts.get("person_attribute_conflicts", 0)
    if conflicts:
        warnings.append(
            f"В {conflicts} строках повторяющийся SYNTHETIC_F имеет другой город/регион. "
            "Для каждого физлица сохранены атрибуты последней строки CSV, "
            "включая пустые значения; все связи с компаниями сохранены."
        )
    if skipped:
        warnings.append(
            f"Пропущено {skipped} более ранних строк с повторяющимся SYNTHETIC_U. "
            "Для каждого ID сохранена только последняя строка CSV."
        )
    return ImportResponse(
        import_id=record.id,
        kind=record.kind,
        filename=record.filename,
        row_count=record.row_count,
        encoding=record.encoding,
        delimiter=record.delimiter,
        reused=reused,
        counts=ImportCounts(**record.counts),
        warnings=warnings,
    )


async def import_csv(
    session: AsyncSession,
    kind: ImportKind,
    parsed: ParsedCsv,
    content: bytes,
    filename: str,
    encoding: CsvEncoding,
    user_id: UUID,
) -> ImportResponse:
    content_hash = sha256(content).hexdigest()
    try:
        await lock_imports(session)
        new_companies = await authorize_import(session, kind, parsed.rows, user_id)
        previous = await find_import(session, kind, content_hash)
        if previous is not None:
            previous.counts = {
                "skipped_duplicate_rows": parsed.skipped_duplicate_rows,
                "person_attribute_conflicts": parsed.person_attribute_conflicts,
                **previous.counts,
            }
            result = response(previous, reused=True)
            await session.commit()
            return result
        record = CsvImport(
            id=uuid4(),
            kind=kind,
            content_hash=content_hash,
            filename=filename,
            encoding=encoding,
            delimiter=parsed.delimiter,
            row_count=len(parsed.rows),
            counts={},
            imported_by_user_id=user_id,
        )
        session.add(record)
        await session.flush()
        save = {"companies": save_companies, "persons": save_persons, "payments": save_payments}[
            kind
        ]
        record.counts = await save(session, parsed.rows, record.id)
        record.counts = {
            **record.counts,
            "skipped_duplicate_rows": parsed.skipped_duplicate_rows,
            "person_attribute_conflicts": parsed.person_attribute_conflicts,
        }
        await join_new_companies(session, new_companies, user_id)
        result = response(record, reused=False)
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise
