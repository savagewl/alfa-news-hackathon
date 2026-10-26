from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

ImportKind = Literal["companies", "persons", "payments"]
CsvEncoding = Literal["utf-8-sig", "cp1251"]


class ImportCounts(BaseModel):
    preserved_person_attributes: int = Field(
        default=0, description="Физлица, чьи прежние атрибуты сохранены из-за чужих связей",
    )
    skipped_inaccessible_rows: int = Field(
        default=0, description="Строки связей физлиц с недоступными компаниями, не импортированные",
    )
    person_attribute_conflicts: int = Field(
        default=0,
        description="Число замен атрибутов повторяющегося SYNTHETIC_F среди импортируемых связей",
    )
    companies: int = 0
    persons: int = 0
    links: int = 0
    payments: int = 0
    placeholder_companies: int = 0
    skipped_duplicate_rows: int = Field(
        default=0,
        description="Число пропущенных более ранних строк компаний с тем же SYNTHETIC_U",
    )


class ImportResponse(BaseModel):
    import_id: UUID
    kind: ImportKind
    filename: str
    row_count: int
    encoding: CsvEncoding
    delimiter: str
    reused: bool
    counts: ImportCounts
    warnings: list[str] = Field(
        default_factory=list,
        description="Предупреждения: например, пропущенные повторы ID компаний",
    )
