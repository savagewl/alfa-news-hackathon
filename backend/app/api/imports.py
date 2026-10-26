from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import CurrentUser, SettingsDep, get_current_user
from app.db.session import SessionDep
from app.repositories.import_access import ImportAccessError
from app.schemas.csv_import import CsvEncoding, ImportKind, ImportResponse
from app.services.csv_import import import_csv
from app.services.csv_parser import CsvValidationError, parse_csv

router = APIRouter(
    prefix="/imports",
    tags=["CSV imports"],
    dependencies=[Depends(get_current_user)],
    responses={
        401: {"description": "Требуется действующий JWT"},
        403: {"description": "CSV затрагивает компании или общие данные вне вашего доступа"},
        413: {"description": "Файл превышает CSV_IMPORT_MAX_BYTES"},
        422: {"description": "Ошибка формата CSV; весь импорт отклонён"},
        503: {"description": "БД недоступна или импорт занят; повторите запрос"},
    },
)

CsvFile = Annotated[UploadFile, File(description="CSV-файл с заголовком")]
Encoding = Annotated[CsvEncoding, Query(description="UTF-8 с/без BOM или Windows-1251")]
Delimiter = Annotated[
    str | None,
    Query(pattern=r"^[,;\t]$", description="Разделитель: запятая, ; или tab; по умолчанию авто"),
]


async def upload(
    kind: ImportKind,
    file: UploadFile,
    encoding: CsvEncoding,
    delimiter: str | None,
    session: SessionDep,
    settings: SettingsDep,
    user: CurrentUser,
) -> ImportResponse:
    try:
        content = await file.read(settings.csv_import_max_bytes + 1)
    finally:
        await file.close()
    if len(content) > settings.csv_import_max_bytes:
        raise HTTPException(status_code=413, detail="CSV file is too large")
    try:
        max_rows = (
            settings.csv_import_payments_max_rows
            if kind == "payments" else settings.csv_import_max_rows
        )
        parsed = await run_in_threadpool(
            parse_csv, content, kind, encoding, delimiter, max_rows
        )
    except CsvValidationError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "invalid_csv", "errors": exc.errors}
        ) from None
    try:
        return await import_csv(
            session,
            kind,
            parsed,
            content,
            file.filename or "upload.csv",
            encoding,
            user.user_id,
        )
    except ImportAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except (SQLAlchemyError, OSError, TimeoutError):
        raise HTTPException(status_code=503, detail="Import unavailable; retry later") from None


@router.post(
    "/companies",
    response_model=ImportResponse,
    summary="Загрузить CSV юридических лиц",
    description=(
        "Все колонки df_Uid_data_hack.csv. Upsert по SYNTHETIC_U. "
        "Заполняет компании-заглушки, созданные импортом связей или платежей. "
        "Новые компании привязываются к текущему пользователю; существующие можно "
        "обновлять только участникам. При повторе SYNTHETIC_U сохраняется только последняя "
        "строка, предыдущие не добавляются; их число — counts.skipped_duplicate_rows, "
        "пояснение — warnings. row_count включает все строки файла. "
        "Весь файл сохраняется одной транзакцией."
    ),
)
async def upload_companies(
    file: CsvFile,
    session: SessionDep,
    settings: SettingsDep,
    user: CurrentUser,
    encoding: Encoding = "utf-8-sig",
    delimiter: Delimiter = None,
) -> ImportResponse:
    return await upload("companies", file, encoding, delimiter, session, settings, user)


@router.post(
    "/persons",
    response_model=ImportResponse,
    summary="Загрузить CSV физлиц и связей с компаниями",
    description=(
        "Колонки SYNTHETIC_F, SYNTHETIC_U, CITY_NAME, RC_NAME. "
        "Физлица и связи обновляются по ID. Все компании из строк должны быть доступны "
        "текущему пользователю. Изменение атрибутов физлица, связанного "
        "с чужой компанией, запрещено."
        " При разных атрибутах одного SYNTHETIC_F сохраняется последняя строка, "
        "все связи с компаниями сохраняются."
    ),
)
async def upload_persons(
    file: CsvFile,
    session: SessionDep,
    settings: SettingsDep,
    user: CurrentUser,
    encoding: Encoding = "utf-8-sig",
    delimiter: Delimiter = None,
) -> ImportResponse:
    return await upload("persons", file, encoding, delimiter, session, settings, user)


@router.post(
    "/payments",
    response_model=ImportResponse,
    summary="Загрузить CSV платёжных связей",
    description=(
        "Колонки PAYER, BENEFICIARY, PAY_AMT, PAY_CNT. Сохраняются агрегаты из строк файла. "
        "Повтор того же файла не дублирует платежи; разные файлы — отдельные наборы, "
        "без неявного суммирования или замены. В каждой строке минимум одна сторона "
        "должна быть вашей компанией. Заглушки контрагентов не дают членства в них."
    ),
)
async def upload_payments(
    file: CsvFile,
    session: SessionDep,
    settings: SettingsDep,
    user: CurrentUser,
    encoding: Encoding = "utf-8-sig",
    delimiter: Delimiter = None,
) -> ImportResponse:
    return await upload("payments", file, encoding, delimiter, session, settings, user)
