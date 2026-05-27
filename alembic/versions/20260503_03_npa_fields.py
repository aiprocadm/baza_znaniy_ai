"""add npa fields to documents

Revision ID: 20260503_03_npa_fields
Revises: 20260503_02_billing_policy_usage
Create Date: 2026-05-03
"""

from alembic import op
import sqlalchemy as sa

revision = "20260503_03_npa_fields"
down_revision = "20260503_02_billing_policy_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("act_type", sa.String(), nullable=True))
    op.add_column("documents", sa.Column("issuer", sa.String(), nullable=True))
    op.add_column("documents", sa.Column("reg_number", sa.String(), nullable=True))
    op.add_column("documents", sa.Column("adoption_date", sa.DateTime(), nullable=True))
    op.add_column("documents", sa.Column("effective_date", sa.DateTime(), nullable=True))
    op.add_column("documents", sa.Column("revision", sa.String(), nullable=True))
    op.add_column(
        "documents", sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true())
    )
    op.create_index("ix_documents_act_type", "documents", ["act_type"])
    op.create_index("ix_documents_issuer", "documents", ["issuer"])
    op.create_index("ix_documents_reg_number", "documents", ["reg_number"])
    op.create_index("ix_documents_adoption_date", "documents", ["adoption_date"])
    op.create_index("ix_documents_effective_date", "documents", ["effective_date"])
    op.create_index("ix_documents_revision", "documents", ["revision"])
    op.create_index("ix_documents_is_active", "documents", ["is_active"])


def downgrade() -> None:
    op.drop_index("ix_documents_is_active", table_name="documents")
    op.drop_index("ix_documents_revision", table_name="documents")
    op.drop_index("ix_documents_effective_date", table_name="documents")
    op.drop_index("ix_documents_adoption_date", table_name="documents")
    op.drop_index("ix_documents_reg_number", table_name="documents")
    op.drop_index("ix_documents_issuer", table_name="documents")
    op.drop_index("ix_documents_act_type", table_name="documents")
    op.drop_column("documents", "is_active")
    op.drop_column("documents", "revision")
    op.drop_column("documents", "effective_date")
    op.drop_column("documents", "adoption_date")
    op.drop_column("documents", "reg_number")
    op.drop_column("documents", "issuer")
    op.drop_column("documents", "act_type")
