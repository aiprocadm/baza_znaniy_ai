"""Initial database schema."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20240919_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column("slug", sa.String(length=100), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("contact_email", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("slug", name="uq_tenants_slug"),
    )
    op.create_index("ix_tenants_slug", "tenants", ["slug"], unique=False)
    op.create_index("ix_tenants_name", "tenants", ["name"], unique=False)

    op.create_table(
        "files",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.String(length=100), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=True),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("retries", sa.Integer(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("chunks", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("tenant_id", "sha256", name="uq_files_tenant_sha"),
    )
    op.create_index("ix_files_document_id", "files", ["document_id"], unique=False)
    op.create_index("ix_files_sha256", "files", ["sha256"], unique=False)
    op.create_index("ix_files_status", "files", ["status"], unique=False)
    op.create_index("ix_files_tenant_id", "files", ["tenant_id"], unique=False)

    op.create_table(
        "pages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("file_id", sa.Integer(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("file_id", "number", name="uq_pages_file_number"),
    )
    op.create_index("ix_pages_file_id", "pages", ["file_id"], unique=False)
    op.create_index("ix_pages_number", "pages", ["number"], unique=False)
    op.create_index("ix_pages_sha256", "pages", ["sha256"], unique=False)

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("page_id", sa.Integer(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("batch", sa.Integer(), nullable=True),
        sa.Column("tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["page_id"], ["pages.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("page_id", "index", name="uq_chunks_page_index"),
        sa.UniqueConstraint("page_id", "sha256", name="uq_chunks_page_sha"),
    )
    op.create_index("ix_chunks_batch", "chunks", ["batch"], unique=False)
    op.create_index("ix_chunks_index", "chunks", ["index"], unique=False)
    op.create_index("ix_chunks_page_id", "chunks", ["page_id"], unique=False)
    op.create_index("ix_chunks_sha256", "chunks", ["sha256"], unique=False)

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_slug", sa.String(length=100), nullable=False),
        sa.Column("file_id", sa.Integer(), nullable=True),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("slug", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("mime_type", sa.String(), nullable=False, server_default="application/octet-stream"),
        sa.Column("status", sa.String(), nullable=False, server_default="queued"),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("chunks", sa.Integer(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["tenant_slug"], ["tenants.slug"], ondelete="CASCADE"),
        sa.UniqueConstraint("sha256", name="uq_documents_sha"),
        sa.UniqueConstraint("tenant_slug", "slug", name="uq_documents_tenant_slug"),
    )
    op.create_index("ix_documents_file_id", "documents", ["file_id"], unique=False)
    op.create_index("ix_documents_sha256", "documents", ["sha256"], unique=False)
    op.create_index("ix_documents_slug", "documents", ["slug"], unique=False)
    op.create_index("ix_documents_status", "documents", ["status"], unique=False)
    op.create_index("ix_documents_tenant_slug", "documents", ["tenant_slug"], unique=False)

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=200), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("tenant_slug", sa.String(length=100), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["tenant_slug"], ["tenants.slug"], ondelete="CASCADE"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_tenant_slug", "users", ["tenant_slug"], unique=False)

    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_slug", sa.String(length=100), nullable=True),
        sa.Column("job_type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("payload", sa.Text(), nullable=True),
        sa.Column("scheduled_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["tenant_slug"], ["tenants.slug"], ondelete="SET NULL"),
    )
    op.create_index("ix_jobs_job_type", "jobs", ["job_type"], unique=False)
    op.create_index("ix_jobs_status", "jobs", ["status"], unique=False)
    op.create_index("ix_jobs_tenant_slug", "jobs", ["tenant_slug"], unique=False)

    op.create_table(
        "settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_slug", sa.String(length=100), nullable=True),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["tenant_slug"], ["tenants.slug"], ondelete="SET NULL"),
        sa.UniqueConstraint("tenant_slug", "key", name="uq_settings_tenant_key"),
    )
    op.create_index("ix_settings_key", "settings", ["key"], unique=False)
    op.create_index("ix_settings_tenant_slug", "settings", ["tenant_slug"], unique=False)

    # SQLite does not support ALTER TABLE ADD CONSTRAINT; these circular-ref
    # FK constraints are only applied on PostgreSQL.
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        op.create_foreign_key(
            "fk_files_document_id",
            "files",
            "documents",
            ["document_id"],
            ["id"],
            ondelete="SET NULL",
        )
        op.create_foreign_key(
            "fk_documents_file_id",
            "documents",
            "files",
            ["file_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        op.drop_constraint("fk_documents_file_id", "documents", type_="foreignkey")
        op.drop_constraint("fk_files_document_id", "files", type_="foreignkey")

    op.drop_index("ix_settings_tenant_slug", table_name="settings")
    op.drop_index("ix_settings_key", table_name="settings")
    op.drop_table("settings")

    op.drop_index("ix_jobs_tenant_slug", table_name="jobs")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_index("ix_jobs_job_type", table_name="jobs")
    op.drop_table("jobs")

    op.drop_index("ix_users_tenant_slug", table_name="users")
    op.drop_table("users")

    op.drop_index("ix_documents_tenant_slug", table_name="documents")
    op.drop_index("ix_documents_status", table_name="documents")
    op.drop_index("ix_documents_slug", table_name="documents")
    op.drop_index("ix_documents_sha256", table_name="documents")
    op.drop_index("ix_documents_file_id", table_name="documents")
    op.drop_table("documents")

    op.drop_index("ix_chunks_sha256", table_name="chunks")
    op.drop_index("ix_chunks_page_id", table_name="chunks")
    op.drop_index("ix_chunks_index", table_name="chunks")
    op.drop_index("ix_chunks_batch", table_name="chunks")
    op.drop_table("chunks")

    op.drop_index("ix_pages_sha256", table_name="pages")
    op.drop_index("ix_pages_number", table_name="pages")
    op.drop_index("ix_pages_file_id", table_name="pages")
    op.drop_table("pages")

    op.drop_index("ix_files_tenant_id", table_name="files")
    op.drop_index("ix_files_status", table_name="files")
    op.drop_index("ix_files_sha256", table_name="files")
    op.drop_index("ix_files_document_id", table_name="files")
    op.drop_table("files")

    op.drop_index("ix_tenants_name", table_name="tenants")
    op.drop_index("ix_tenants_slug", table_name="tenants")
    op.drop_table("tenants")
