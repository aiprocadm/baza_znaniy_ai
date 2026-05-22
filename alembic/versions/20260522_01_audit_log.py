"""Add audit_log table for security and request auditing.

Revision ID: 20260522_01_audit_log
Revises: 20260506_01_api_keys_usage_rag
Create Date: 2026-05-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260522_01_audit_log"
down_revision = "20260506_01_api_keys_usage_rag"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime, nullable=False, index=True),
        sa.Column("event", sa.String(64), nullable=False, index=True),
        sa.Column("user_id", sa.String(64), nullable=True, index=True),
        sa.Column("tenant", sa.String(64), nullable=True, index=True),
        sa.Column("ip", sa.String(45), nullable=True),  # IPv6 max length
        sa.Column("request_path", sa.String(512), nullable=True),
        sa.Column("request_method", sa.String(8), nullable=True),
        sa.Column("status_code", sa.Integer, nullable=True),
        sa.Column("payload_json", sa.Text, nullable=True),
        sa.Column("correlation_id", sa.String(64), nullable=True, index=True),
    )


def downgrade() -> None:
    op.drop_table("audit_log")
