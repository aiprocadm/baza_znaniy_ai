"""Align data model with target KB schema."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260503_01"
down_revision = "20240919_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_table("settings")
    op.drop_table("jobs")
    op.drop_table("pages")
    op.drop_constraint("fk_documents_file_id", "documents", type_="foreignkey")
    op.drop_constraint("fk_files_document_id", "files", type_="foreignkey")
    op.drop_table("files")

    op.alter_column("tenants", "slug", new_column_name="legacy_slug")
    op.add_column("tenants", sa.Column("id", sa.Integer(), autoincrement=True, nullable=True))
    op.execute("UPDATE tenants SET id = row_number() over (order by legacy_slug)")
    op.alter_column("tenants", "id", nullable=False)
    op.create_primary_key("pk_tenants", "tenants", ["id"])
    op.add_column("tenants", sa.Column("slug", sa.String(length=100), nullable=True))
    op.execute("UPDATE tenants SET slug = legacy_slug")
    op.alter_column("tenants", "slug", nullable=False)
    op.create_unique_constraint("uq_tenants_slug", "tenants", ["slug"])
    op.add_column(
        "tenants",
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
    )

    op.add_column("users", sa.Column("tenant_id", sa.Integer(), nullable=True))
    op.execute(
        "UPDATE users u SET tenant_id = t.id FROM tenants t WHERE u.tenant_slug = t.legacy_slug"
    )
    op.alter_column("users", "tenant_id", nullable=False)
    op.drop_constraint("users_tenant_slug_fkey", "users", type_="foreignkey")
    op.drop_column("users", "tenant_slug")
    op.create_foreign_key(
        "fk_users_tenant_id", "users", "tenants", ["tenant_id"], ["id"], ondelete="CASCADE"
    )

    op.add_column("documents", sa.Column("tenant_id", sa.Integer(), nullable=True))
    op.execute(
        "UPDATE documents d SET tenant_id = t.id FROM tenants t WHERE d.tenant_slug = t.legacy_slug"
    )
    op.alter_column("documents", "tenant_id", nullable=False)
    op.drop_constraint("documents_tenant_slug_fkey", "documents", type_="foreignkey")
    op.drop_column("documents", "tenant_slug")
    op.drop_column("documents", "sha256")
    op.drop_column("documents", "mime_type")
    op.drop_column("documents", "error")
    op.drop_column("documents", "chunks")
    op.drop_column("documents", "content")
    op.add_column("documents", sa.Column("owner_user_id", sa.Integer(), nullable=True))
    op.add_column(
        "documents", sa.Column("chunks_count", sa.Integer(), server_default="0", nullable=False)
    )
    op.create_foreign_key(
        "fk_documents_tenant_id", "documents", "tenants", ["tenant_id"], ["id"], ondelete="CASCADE"
    )
    op.create_foreign_key(
        "fk_documents_owner_user_id",
        "documents",
        "users",
        ["owner_user_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_table(
        "document_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("content_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("document_id", "version", name="uq_document_versions_document_version"),
    )

    op.add_column("chunks", sa.Column("tenant_id", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("document_id", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("document_version_id", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("chunk_index", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("vector_backend", sa.String(length=64), nullable=True))
    op.add_column("chunks", sa.Column("vector_id", sa.String(length=255), nullable=True))
    op.drop_column("chunks", "page_id")
    op.drop_column("chunks", "index")
    op.drop_column("chunks", "sha256")
    op.drop_column("chunks", "batch")
    op.alter_column("chunks", "chunk_index", nullable=False)
    op.create_foreign_key(
        "fk_chunks_tenant_id", "chunks", "tenants", ["tenant_id"], ["id"], ondelete="CASCADE"
    )
    op.create_foreign_key(
        "fk_chunks_document_id", "chunks", "documents", ["document_id"], ["id"], ondelete="CASCADE"
    )
    op.create_foreign_key(
        "fk_chunks_document_version_id",
        "chunks",
        "document_versions",
        ["document_version_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_table(
        "subscriptions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("plan_code", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("current_period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
    )
    op.create_table(
        "billing_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("subscription_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("amount_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("currency", sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subscription_id"], ["subscriptions.id"], ondelete="SET NULL"),
    )


def downgrade() -> None:
    op.drop_table("billing_events")
    op.drop_table("subscriptions")
    op.drop_table("document_versions")
