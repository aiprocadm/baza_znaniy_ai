"""Add plans and usage counters for policy engine."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260503_02"
down_revision = "20260503_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "plans",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("code", sa.String(length=64), nullable=False, unique=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("max_storage_bytes", sa.BigInteger(), nullable=False, server_default="52428800"),
        sa.Column("max_documents", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("max_search_requests", sa.Integer(), nullable=False, server_default="1000"),
        sa.Column("max_llm_requests", sa.Integer(), nullable=False, server_default="1000"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "usage_counters",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("storage_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("documents_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("search_requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("llm_requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("tenant_id", "period_start", "period_end", name="uq_usage_counter_period"),
    )


def downgrade() -> None:
    op.drop_table("usage_counters")
    op.drop_table("plans")
