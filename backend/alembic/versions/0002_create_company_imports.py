"""Company data, persons, links, payment aggregates and idempotent CSV imports."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0002_create_company_imports"
down_revision = "0001_create_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "csv_imports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("encoding", sa.Text(), nullable=False),
        sa.Column("delimiter", sa.Text(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("counts", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_csv_imports"),
        sa.UniqueConstraint("kind", "content_hash", name="uq_csv_imports_kind"),
    )
    company_text_fields = [
        "group_id",
        "org_type",
        "ownership_form",
        "age_category",
        "development_stage",
        "okved_code",
        "okved_name",
        "employees_fns_category",
        "employees_spark_category",
        "client_type",
        "fts_country_name",
        "city",
        "region",
    ]
    company_flags = [
        "key_owner_flag",
        "has_rko",
        "has_guarantees",
        "has_credit",
        "has_leasing",
        "has_deposit",
        "has_payroll",
        "has_fts_data",
        "has_factoring",
    ]
    op.create_table(
        "companies",
        sa.Column("id", sa.Text(), nullable=False),
        *[sa.Column(name, sa.Text(), nullable=True) for name in company_text_fields],
        *[sa.Column(name, sa.Boolean(), nullable=True) for name in company_flags],
        sa.Column("annual_revenue", sa.Numeric(), nullable=True),
        sa.Column("source_data", postgresql.JSONB(), nullable=True),
        sa.Column("is_placeholder", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("last_import_id", sa.Uuid(), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_companies"),
        sa.ForeignKeyConstraint(
            ["last_import_id"], ["csv_imports.id"], name="fk_companies_last_import_id_csv_imports"
        ),
    )
    op.create_index("ix_companies_group_id", "companies", ["group_id"])
    op.create_table(
        "persons",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("city", sa.Text(), nullable=True),
        sa.Column("region", sa.Text(), nullable=True),
        sa.Column("last_import_id", sa.Uuid(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_persons"),
        sa.ForeignKeyConstraint(
            ["last_import_id"], ["csv_imports.id"], name="fk_persons_last_import_id_csv_imports"
        ),
    )
    op.create_table(
        "company_person_links",
        sa.Column("company_id", sa.Text(), nullable=False),
        sa.Column("person_id", sa.Text(), nullable=False),
        sa.Column("last_import_id", sa.Uuid(), nullable=False),
        sa.PrimaryKeyConstraint("company_id", "person_id", name="pk_company_person_links"),
        sa.ForeignKeyConstraint(
            ["company_id"], ["companies.id"], name="fk_company_person_links_company_id_companies"
        ),
        sa.ForeignKeyConstraint(
            ["person_id"], ["persons.id"], name="fk_company_person_links_person_id_persons"
        ),
        sa.ForeignKeyConstraint(
            ["last_import_id"],
            ["csv_imports.id"],
            name="fk_company_person_links_last_import_id_csv_imports",
        ),
    )
    op.create_index("ix_company_person_links_person_id", "company_person_links", ["person_id"])
    op.create_table(
        "company_payments",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("payer_company_id", sa.Text(), nullable=False),
        sa.Column("beneficiary_company_id", sa.Text(), nullable=False),
        sa.Column("payment_amount", sa.Numeric(), nullable=False),
        sa.Column("payment_count", sa.BigInteger(), nullable=False),
        sa.Column("import_id", sa.Uuid(), nullable=False),
        sa.Column("source_row_number", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_company_payments"),
        sa.UniqueConstraint("import_id", "source_row_number", name="uq_company_payments_import_id"),
        sa.CheckConstraint("payment_amount >= 0", name="ck_company_payments_nonnegative_amount"),
        sa.CheckConstraint("payment_count >= 0", name="ck_company_payments_nonnegative_count"),
        sa.ForeignKeyConstraint(
            ["payer_company_id"],
            ["companies.id"],
            name="fk_company_payments_payer_company_id_companies",
        ),
        sa.ForeignKeyConstraint(
            ["beneficiary_company_id"],
            ["companies.id"],
            name="fk_company_payments_beneficiary_company_id_companies",
        ),
        sa.ForeignKeyConstraint(
            ["import_id"], ["csv_imports.id"], name="fk_company_payments_import_id_csv_imports"
        ),
    )
    op.create_index(
        "ix_company_payments_payer_company_id", "company_payments", ["payer_company_id"]
    )
    op.create_index(
        "ix_company_payments_beneficiary_company_id", "company_payments", ["beneficiary_company_id"]
    )


def downgrade() -> None:
    op.drop_table("company_payments")
    op.drop_table("company_person_links")
    op.drop_table("persons")
    op.drop_table("companies")
    op.drop_table("csv_imports")
