"""User-company membership; existing companies stay inaccessible until explicitly assigned."""

import sqlalchemy as sa

from alembic import op

revision = "0003_company_memberships"
down_revision = "0002_create_company_imports"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("companies", sa.Column("name", sa.Text(), nullable=True))
    op.add_column("csv_imports", sa.Column("imported_by_user_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_csv_imports_imported_by_user_id_users",
        "csv_imports",
        "users",
        ["imported_by_user_id"],
        ["user_id"],
    )
    op.create_table(
        "company_memberships",
        sa.Column("company_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("added_by_user_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("company_id", "user_id", name="pk_company_memberships"),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name="fk_company_memberships_company_id_companies",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            name="fk_company_memberships_user_id_users",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["added_by_user_id"],
            ["users.user_id"],
            name="fk_company_memberships_added_by_user_id_users",
        ),
    )
    op.create_index("ix_company_memberships_user_id", "company_memberships", ["user_id"])


def downgrade() -> None:
    op.drop_table("company_memberships")
    op.drop_constraint(
        "fk_csv_imports_imported_by_user_id_users", "csv_imports", type_="foreignkey"
    )
    op.drop_column("csv_imports", "imported_by_user_id")
    op.drop_column("companies", "name")
