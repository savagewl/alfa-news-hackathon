"""Durable AI workers, profile cache and contextual news chat.

Revision ID: 0008_ai_background_state
Revises: 0007_merge_news_branches
"""

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0008_ai_background_state"
down_revision = "0007_merge_news_branches"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # news_embeddings also serves as a durable queue. Pending rows intentionally
    # have no summary/vector until both remote stages finish.
    op.alter_column("news_embeddings", "summary", existing_type=sa.Text(), nullable=True)
    op.alter_column(
        "news_embeddings", "vector", existing_type=Vector(1024), nullable=True
    )
    op.add_column(
        "news_embeddings",
        sa.Column("source_content_hash", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "news_embeddings",
        sa.Column("input_fingerprint", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "news_embeddings",
        sa.Column("llm_model_version", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "news_embeddings",
        sa.Column("prompt_version", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "news_embeddings",
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
    )
    op.add_column(
        "news_embeddings",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "news_embeddings", sa.Column("next_attempt_at", sa.DateTime(timezone=True))
    )
    op.add_column("news_embeddings", sa.Column("last_error", sa.Text()))
    op.add_column("news_embeddings", sa.Column("lease_token", sa.Uuid()))
    op.add_column("news_embeddings", sa.Column("lease_until", sa.DateTime(timezone=True)))
    # Existing rows do not carry enough provenance to distinguish mock and real
    # output, so the worker must refresh them before readers may use them.
    op.execute("UPDATE news_embeddings SET status = 'pending'")
    op.create_index(
        "ix_news_embeddings_work",
        "news_embeddings",
        ["status", "next_attempt_at", "lease_until"],
    )

    op.create_table(
        "company_profile_cache",
        sa.Column(
            "company_id",
            sa.Text(),
            sa.ForeignKey("companies.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("input_fingerprint", sa.Text(), nullable=False),
        sa.Column("profile_text", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("fields", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("vectors", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("embedding_model_version", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("last_error", sa.Text()),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_company_profile_cache_work",
        "company_profile_cache",
        ["status", "next_attempt_at"],
    )

    for name in (
        "company_input_fingerprint",
        "news_input_fingerprint",
        "request_fingerprint",
        "retrieval_fingerprint",
        "model_prompt_version",
    ):
        op.add_column("news_interpretations", sa.Column(name, sa.Text()))
    op.create_unique_constraint(
        "uq_news_interpretations_request_fingerprint",
        "news_interpretations",
        ["request_fingerprint"],
    )

    op.create_table(
        "news_chat_threads",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(),
            sa.ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "company_id", sa.Text(),
            sa.ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "news_id", sa.Uuid(),
            sa.ForeignKey("news_articles.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now()
        ),
    )
    op.create_index(
        "ix_news_chat_threads_owner_context",
        "news_chat_threads",
        ["user_id", "company_id", "news_id", "updated_at"],
    )

    op.create_table(
        "news_chat_messages",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "thread_id", sa.Uuid(),
            sa.ForeignKey("news_chat_threads.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("client_message_id", sa.Uuid()),
        sa.Column(
            "user_id", sa.Uuid(),
            sa.ForeignKey("users.user_id", ondelete="SET NULL")
        ),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("answer_kind", sa.Text()),
        sa.Column(
            "reply_to_id", sa.Uuid(),
            sa.ForeignKey("news_chat_messages.id", ondelete="CASCADE"), unique=True
        ),
        sa.Column("sources", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("warnings", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("status", sa.Text(), nullable=False, server_default="ready"),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "thread_id", "client_message_id",
            name="uq_news_chat_messages_thread_id"
        ),
    )
    op.create_index(
        "ix_news_chat_messages_thread_order",
        "news_chat_messages",
        ["thread_id", "created_at", "id"],
    )


def downgrade() -> None:
    op.drop_index("ix_news_chat_messages_thread_order", table_name="news_chat_messages")
    op.drop_table("news_chat_messages")
    op.drop_index("ix_news_chat_threads_owner_context", table_name="news_chat_threads")
    op.drop_table("news_chat_threads")

    op.drop_constraint(
        "uq_news_interpretations_request_fingerprint",
        table_name="news_interpretations",
        type_="unique",
    )
    for name in reversed((
        "company_input_fingerprint",
        "news_input_fingerprint",
        "request_fingerprint",
        "retrieval_fingerprint",
        "model_prompt_version",
    )):
        op.drop_column("news_interpretations", name)

    op.drop_index("ix_company_profile_cache_work", table_name="company_profile_cache")
    op.drop_table("company_profile_cache")

    op.drop_index("ix_news_embeddings_work", table_name="news_embeddings")
    for name in (
        "lease_until", "lease_token", "last_error", "next_attempt_at", "attempts",
        "status", "prompt_version", "llm_model_version", "input_fingerprint",
        "source_content_hash",
    ):
        op.drop_column("news_embeddings", name)
    op.execute("DELETE FROM news_embeddings WHERE summary IS NULL OR vector IS NULL")
    op.alter_column(
        "news_embeddings", "vector", existing_type=Vector(1024), nullable=False
    )
    op.alter_column("news_embeddings", "summary", existing_type=sa.Text(), nullable=False)
