from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.schemas.scalars import DecimalString

PaymentDirection = Literal["all", "incoming", "outgoing"]


class Page[T](BaseModel):
    items: list[T]
    total: int = Field(description="Число записей по заданным фильтрам до пагинации")
    limit: int
    offset: int


class CompanyListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str = Field(
        description="SYNTHETIC_U или ID создания; ведущие нули сохраняются", examples=["001"]
    )
    name: str | None = Field(description="Название, заданное при создании; в CSV его нет")
    group_id: str | None
    okved_code: str | None
    okved_name: str | None
    city: str | None = Field(examples=["Казань"])
    region: str | None = Field(examples=["Татарстан"])
    annual_revenue: DecimalString | None = Field(
        description=(
            "Выручка из REVENUE_COMPANY; точная десятичная строка без экспоненциальной записи"
        ),
        examples=["15000000000.50"],
    )
    is_placeholder: bool = Field(
        description="Компания известна только по связи/платежу; CSV ещё не загружен"
    )


class CompanyResponse(CompanyListItem):
    org_type: str | None
    ownership_form: str | None
    age_category: str | None
    development_stage: str | None
    key_owner_flag: bool | None
    employees_fns_category: str | None
    employees_spark_category: str | None
    client_type: str | None
    has_rko: bool | None
    has_guarantees: bool | None
    has_credit: bool | None
    has_leasing: bool | None
    has_deposit: bool | None
    has_payroll: bool | None
    has_fts_data: bool | None
    fts_country_name: str | None
    has_factoring: bool | None
    source_data: dict[str, str] | None = Field(
        description="Исходная строка CSV, включая дополнительные колонки"
    )
    last_import_id: UUID | None
    updated_at: datetime


class CompanyPersonResponse(BaseModel):
    person_id: str = Field(description="Обезличенный SYNTHETIC_F")
    city: str | None
    region: str | None
    person_import_id: UUID = Field(description="Последний импорт атрибутов физлица")
    link_import_id: UUID = Field(description="Последний импорт связи физлица с выбранной компанией")


class CompanyPaymentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    payer_company_id: str = Field(description="PAYER — отправитель")
    beneficiary_company_id: str = Field(description="BENEFICIARY — получатель")
    payment_amount: DecimalString = Field(
        description="PAY_AMT; точная десятичная строка без экспоненциальной записи",
        examples=["123.45"],
    )
    payment_count: int = Field(description="PAY_CNT — количество платежей в исходной строке")
    import_id: UUID = Field(description="Набор данных: разные импорты могут перекрываться")
    source_row_number: int


class CompanyPaymentPage(Page[CompanyPaymentResponse]):
    direction: PaymentDirection
    import_id: UUID | None = Field(
        description="Применённый фильтр; null означает строки всех импортов"
    )


class PaymentImportResponse(BaseModel):
    import_id: UUID
    filename: str
    created_at: datetime
    row_count: int = Field(description="Число строк во всём импортированном файле")
    company_payment_rows: int = Field(description="Число строк данного импорта с участием компании")


class CompanyApiError(BaseModel):
    detail: str


ShortText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=512)]


class CreateCompanyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
        description="Новый ID / SYNTHETIC_U. Если не задан, генерируется UUID в текстовом виде",
    )
    name: str = Field(min_length=1, max_length=255, examples=["ООО Пример"])
    group_id: ShortText | None = None
    org_type: ShortText | None = None
    ownership_form: ShortText | None = None
    age_category: ShortText | None = None
    development_stage: ShortText | None = None
    okved_code: ShortText | None = None
    okved_name: ShortText | None = None
    key_owner_flag: bool | None = None
    employees_fns_category: ShortText | None = None
    employees_spark_category: ShortText | None = None
    client_type: ShortText | None = None
    has_rko: bool | None = None
    has_guarantees: bool | None = None
    has_credit: bool | None = None
    has_leasing: bool | None = None
    has_deposit: bool | None = None
    has_payroll: bool | None = None
    has_fts_data: bool | None = None
    fts_country_name: ShortText | None = None
    has_factoring: bool | None = None
    annual_revenue: Decimal | None = Field(default=None, ge=0, max_digits=101, decimal_places=20)
    city: ShortText | None = None
    region: ShortText | None = None


class AddCompanyUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UUID = Field(description="userId существующего пользователя из /user/register")


class CompanyMembershipResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    company_id: str
    user_id: UUID
    added_by_user_id: UUID | None
    created_at: datetime


class CompanyUserResponse(BaseModel):
    user_id: UUID
    full_name: str | None
    joined_at: datetime
