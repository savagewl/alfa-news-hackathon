import json
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
from app.repositories.import_access import (
    accessible_person_links,
    authorize_import,
    join_new_companies,
)
from app.schemas.csv_import import CsvEncoding, ImportCounts, ImportKind, ImportResponse
from app.services.csv_parser import ParsedCsv


def response(record: CsvImport, *, reused: bool) -> ImportResponse:
    skipped = record.counts.get("skipped_duplicate_rows", 0)
    warnings = []
    preserved = record.counts.get("preserved_person_attributes", 0)
    if preserved:
        warnings.append(
            f"У {preserved} физлиц сохранены прежние атрибуты: они связаны с недоступными "
            "компаниями. Все импортируемые связи сохранены без выдачи доступа к компаниям."
        )
    inaccessible = record.counts.get("skipped_inaccessible_rows", 0)
    if inaccessible:
        warnings.append(
            f"Пропущено {inaccessible} строк связей с недоступными компаниями. "
            "Импортированы только связи с доступными компаниями. "
            "Для остальных сначала загрузите компании или получите доступ к ним."
        )
    conflicts = record.counts.get("person_attribute_conflicts", 0)
    if conflicts:
        warnings.append(
            f"В {conflicts} строках повторяющийся SYNTHETIC_F имеет другой город/регион. "
            "Для физлиц без защищённых атрибутов сохранена последняя строка CSV, "
            "включая пустые значения; связи из импортированной части файла сохранены."
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
    *,
    skip_inaccessible_person_links: bool = False,
) -> ImportResponse:
    content_hash = sha256(content).hexdigest()
    if kind == "persons":
        # Reprocess files previously imported only for accessible companies.
        content_hash = sha256(f"persons-links-v2:{user_id}:{content_hash}".encode()).hexdigest()
    try:
        await lock_imports(session)
        rows = parsed.rows
        skipped_inaccessible_rows = 0
        conflicts = parsed.person_attribute_conflicts
        if kind == "persons" and skip_inaccessible_person_links:
            rows = await accessible_person_links(session, rows, user_id)
            skipped_inaccessible_rows = len(parsed.rows) - len(rows)
            if skipped_inaccessible_rows:
                # A filtered import must not reuse a different user's/access subset's result.
                scope = json.dumps([
                    content_hash, str(user_id), sorted({row["company_id"] for row in rows})
                ], ensure_ascii=False, separators=(",", ":"))
                content_hash = sha256(scope.encode()).hexdigest()
                seen = {}
                conflicts = 0
                for row in rows:
                    attributes = (row["city"], row["region"])
                    if row["id"] in seen and seen[row["id"]] != attributes:
                        conflicts += 1
                    seen[row["id"]] = attributes
        new_companies = await authorize_import(session, kind, rows, user_id)
        previous = await find_import(session, kind, content_hash)
        if previous is not None:
            previous.counts = {
                "skipped_duplicate_rows": parsed.skipped_duplicate_rows,
                "person_attribute_conflicts": conflicts,
                "skipped_inaccessible_rows": skipped_inaccessible_rows,
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
        if kind == "persons":
            record.counts = await save_persons(session, rows, record.id, user_id=user_id)
        else:
            record.counts = await save(session, rows, record.id)
        record.counts = {
            **record.counts,
            "skipped_duplicate_rows": parsed.skipped_duplicate_rows,
            "person_attribute_conflicts": conflicts,
            "skipped_inaccessible_rows": skipped_inaccessible_rows,
        }
        await join_new_companies(session, new_companies, user_id)
        result = response(record, reused=False)
        await session.commit()
        return result
    except Exception:
        await session.rollback()
        raise
