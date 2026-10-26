from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import CurrentUser, get_current_user
from app.db.session import SessionDep
from app.repositories.company import (
    CompanyExistsError,
    CompanyNotFoundError,
    CompanyRepository,
    MemberUserNotFoundError,
)
from app.schemas.company import (
    AddCompanyUserRequest,
    CompanyApiError,
    CompanyListItem,
    CompanyMembershipResponse,
    CompanyPaymentPage,
    CompanyPersonResponse,
    CompanyResponse,
    CompanyUserResponse,
    CreateCompanyRequest,
    Page,
    PaymentDirection,
    PaymentImportResponse,
)

router = APIRouter(
    prefix="/companies",
    tags=["Companies"],
    dependencies=[Depends(get_current_user)],
    responses={
        401: {"model": CompanyApiError, "description": "Требуется действующий JWT"},
        503: {"model": CompanyApiError, "description": "База данных недоступна"},
    },
)
NOT_FOUND = {
    404: {
        "model": CompanyApiError,
        "description": "Компания не найдена или текущий пользователь не состоит в ней",
        "content": {"application/json": {"example": {"detail": "Company not found"}}},
    }
}
CompanyId = Annotated[
    str,
    Path(
        min_length=1,
        max_length=512,
        description="ID компании: SYNTHETIC_U из CSV или ID из POST /companies",
        examples=["001"],
    ),
]
Limit = Annotated[int, Query(ge=1, le=100, description="Размер страницы, от 1 до 100")]
Offset = Annotated[int, Query(ge=0, description="Число записей, пропускаемых в начале выборки")]


async def get_repository(
    session: SessionDep, user: CurrentUser
) -> AsyncIterator[CompanyRepository]:
    try:
        yield CompanyRepository(session, user.user_id)
    except CompanyNotFoundError:
        await session.rollback()
        raise HTTPException(status_code=404, detail="Company not found") from None
    except CompanyExistsError:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Company ID already exists") from None
    except MemberUserNotFoundError:
        await session.rollback()
        raise HTTPException(status_code=404, detail="User not found") from None
    except (SQLAlchemyError, OSError, TimeoutError):
        await session.rollback()
        raise HTTPException(status_code=503, detail="Database unavailable") from None


Repository = Annotated[CompanyRepository, Depends(get_repository)]


@router.get(
    "",
    response_model=Page[CompanyListItem],
    summary="Получить список компаний",
    description=(
        "Только компании, в которых состоит текущий пользователь. "
        "Возвращает краткие карточки, включая заглушки из платежей/связей. "
        "Фильтры объединяются через AND. Сортировка по ID как тексту, по возрастанию. "
        "Неизвестные значения остаются null."
    ),
)
async def list_companies(
    repository: Repository,
    limit: Limit = 20,
    offset: Offset = 0,
    q: Annotated[
        str | None,
        Query(
            min_length=1,
            max_length=200,
            description=("Подстрока ID, названия компании, ОКВЭД или города; % и _ буквальные"),
        ),
    ] = None,
    group_id: Annotated[
        str | None, Query(max_length=512, description="Точное совпадение GROUP_ID")
    ] = None,
    region: Annotated[
        str | None, Query(max_length=512, description="Точное совпадение REGION")
    ] = None,
    is_placeholder: Annotated[
        bool | None, Query(description="Отбор только заглушок или полных записей")
    ] = None,
) -> dict:
    return await repository.list(
        limit=limit,
        offset=offset,
        q=q,
        group_id=group_id,
        region=region,
        is_placeholder=is_placeholder,
    )


@router.get(
    "/{company_id}",
    response_model=CompanyResponse,
    summary="Получить полную карточку компании",
    description=(
        "Все нормализованные поля CSV, исходная строка source_data и метаданные импорта. "
        "Требуется членство в компании. Для доступной компании-заглушки возвращается 200 с "
        "is_placeholder=true и неизвестными атрибутами null. AI-summary здесь не формируется."
    ),
    responses=NOT_FOUND,
)
async def get_company(company_id: CompanyId, repository: Repository):
    return await repository.get(company_id)


@router.get(
    "/{company_id}/persons",
    response_model=Page[CompanyPersonResponse],
    summary="Получить связанных с компанией физлиц",
    description=(
        "Обезличенные SYNTHETIC_F, город/регион и происхождение атрибутов и связи. "
        "Сортировка по ID физлица. Тип связи (владелец, директор и т. п.) "
        "не задан в исходных данных и не добавляется. Для компании без связей — пустой items."
    ),
    responses=NOT_FOUND,
)
async def get_company_persons(
    company_id: CompanyId,
    repository: Repository,
    limit: Limit = 20,
    offset: Offset = 0,
) -> dict:
    return await repository.persons(company_id, limit=limit, offset=offset)


@router.get(
    "/{company_id}/payments",
    response_model=CompanyPaymentPage,
    summary="Получить платёжные связи компании",
    description=(
        "Строки агрегатов PAY_AMT/PAY_CNT, не отдельные транзакции. "
        "direction: incoming — компания получатель, outgoing — отправитель, all — оба направления. "
        "Самоплатёж при all возвращается один раз. Без import_id возвращаются строки всех наборов "
        "без суммирования: разные файлы могут перекрываться. Для анализа выберите import_id "
        "из /companies/{company_id}/payment-imports. Период и валюта неизвестны. "
        "Сортировка по ID записи. Неизвестный import_id даёт пустую выборку."
    ),
    responses=NOT_FOUND,
)
async def get_company_payments(
    company_id: CompanyId,
    repository: Repository,
    limit: Limit = 20,
    offset: Offset = 0,
    direction: Annotated[
        PaymentDirection, Query(description="Направление относительно компании")
    ] = "all",
    import_id: Annotated[
        UUID | None, Query(description="ID набора из импорта CSV платежей")
    ] = None,
) -> dict:
    return await repository.payments(
        company_id,
        limit=limit,
        offset=offset,
        direction=direction,
        import_id=import_id,
    )


@router.get(
    "/{company_id}/payment-imports",
    response_model=Page[PaymentImportResponse],
    summary="Получить наборы платежей с участием компании",
    description=(
        "Позволяет выбрать import_id для получения платежей из одного файла. "
        "Сортировка по времени импорта, затем UUID, по убыванию. "
        "Дата импорта не является датой платежей. row_count относится ко всему файлу, "
        "company_payment_rows — только к выбранной компании."
    ),
    responses=NOT_FOUND,
)
async def get_company_payment_imports(
    company_id: CompanyId,
    repository: Repository,
    limit: Limit = 20,
    offset: Offset = 0,
) -> dict:
    return await repository.payment_imports(company_id, limit=limit, offset=offset)


@router.post(
    "",
    response_model=CompanyResponse,
    status_code=201,
    summary="Создать компанию",
    description=(
        "Создаёт компанию и атомарно добавляет текущего пользователя в её участники. "
        "name обязателен. ID можно задать для будущего CSV (SYNTHETIC_U), "
        "иначе генерируется UUID. Существующую компанию присвоить себе нельзя — 409. "
        "Владение реальным юридическим лицом в прототипе не проверяется."
    ),
    responses={409: {"model": CompanyApiError, "description": "ID компании уже занят"}},
)
async def create_company(body: CreateCompanyRequest, repository: Repository):
    return await repository.create(body)


@router.post(
    "/{company_id}/users",
    response_model=CompanyMembershipResponse,
    summary="Добавить пользователя в компанию по ID",
    description=(
        "Доступно любому участнику компании; ролей нет. user_id — UUID уже "
        "зарегистрированного пользователя. После добавления он получает доступ к компании "
        "и её данным. Повторное добавление возвращает существующую связь без дубля. "
        "Нельзя добавить себя или другого пользователя в чужую компанию."
    ),
    responses={
        404: {
            "model": CompanyApiError,
            "description": "Компания недоступна либо добавляемый пользователь не найден",
        }
    },
)
async def add_company_user(
    company_id: CompanyId,
    body: AddCompanyUserRequest,
    repository: Repository,
):
    return await repository.add_user(company_id, body.user_id)


@router.get(
    "/{company_id}/users",
    response_model=Page[CompanyUserResponse],
    summary="Получить участников компании",
    description=(
        "Только для участника компании. ID, отображаемое имя и дата вступления; "
        "email и данные авторизации не возвращаются. Сортировка по UUID пользователя."
    ),
    responses=NOT_FOUND,
)
async def get_company_users(
    company_id: CompanyId,
    repository: Repository,
    limit: Limit = 20,
    offset: Offset = 0,
) -> dict:
    return await repository.users(company_id, limit=limit, offset=offset)
