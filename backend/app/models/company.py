from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CsvImport(Base):
    __tablename__ = "csv_imports"
    __table_args__ = (UniqueConstraint("kind", "content_hash"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    kind: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(Text)
    filename: Mapped[str] = mapped_column(Text)
    encoding: Mapped[str] = mapped_column(Text)
    delimiter: Mapped[str] = mapped_column(Text)
    row_count: Mapped[int] = mapped_column(Integer)
    counts: Mapped[dict] = mapped_column(JSONB)
    imported_by_user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.user_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    group_id: Mapped[str | None] = mapped_column(Text, index=True)
    org_type: Mapped[str | None] = mapped_column(Text)
    ownership_form: Mapped[str | None] = mapped_column(Text)
    age_category: Mapped[str | None] = mapped_column(Text)
    development_stage: Mapped[str | None] = mapped_column(Text)
    okved_code: Mapped[str | None] = mapped_column(Text)
    okved_name: Mapped[str | None] = mapped_column(Text)
    key_owner_flag: Mapped[bool | None] = mapped_column(Boolean)
    employees_fns_category: Mapped[str | None] = mapped_column(Text)
    employees_spark_category: Mapped[str | None] = mapped_column(Text)
    client_type: Mapped[str | None] = mapped_column(Text)
    has_rko: Mapped[bool | None] = mapped_column(Boolean)
    has_guarantees: Mapped[bool | None] = mapped_column(Boolean)
    has_credit: Mapped[bool | None] = mapped_column(Boolean)
    has_leasing: Mapped[bool | None] = mapped_column(Boolean)
    has_deposit: Mapped[bool | None] = mapped_column(Boolean)
    has_payroll: Mapped[bool | None] = mapped_column(Boolean)
    has_fts_data: Mapped[bool | None] = mapped_column(Boolean)
    fts_country_name: Mapped[str | None] = mapped_column(Text)
    has_factoring: Mapped[bool | None] = mapped_column(Boolean)
    annual_revenue: Mapped[Decimal | None] = mapped_column(Numeric)
    city: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str | None] = mapped_column(Text)
    source_data: Mapped[dict | None] = mapped_column(JSONB)
    is_placeholder: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    last_import_id: Mapped[UUID | None] = mapped_column(ForeignKey("csv_imports.id"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CompanyMembership(Base):
    __tablename__ = "company_memberships"

    company_id: Mapped[str] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True, index=True
    )
    added_by_user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.user_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    city: Mapped[str | None] = mapped_column(Text)
    region: Mapped[str | None] = mapped_column(Text)
    last_import_id: Mapped[UUID] = mapped_column(ForeignKey("csv_imports.id"))


class CompanyPersonLink(Base):
    __tablename__ = "company_person_links"

    company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), primary_key=True)
    person_id: Mapped[str] = mapped_column(ForeignKey("persons.id"), primary_key=True, index=True)
    last_import_id: Mapped[UUID] = mapped_column(ForeignKey("csv_imports.id"))


class CompanyPayment(Base):
    __tablename__ = "company_payments"
    __table_args__ = (
        UniqueConstraint("import_id", "source_row_number"),
        CheckConstraint("payment_amount >= 0", name="nonnegative_amount"),
        CheckConstraint("payment_count >= 0", name="nonnegative_count"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    payer_company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), index=True)
    beneficiary_company_id: Mapped[str] = mapped_column(ForeignKey("companies.id"), index=True)
    payment_amount: Mapped[Decimal] = mapped_column(Numeric)
    payment_count: Mapped[int] = mapped_column(BigInteger)
    import_id: Mapped[UUID] = mapped_column(ForeignKey("csv_imports.id"))
    source_row_number: Mapped[int] = mapped_column(Integer)
