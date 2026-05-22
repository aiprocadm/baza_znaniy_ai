"""Add PDF citation columns: page_number, has_original_file, file_relpath.

Revision ID: 20260522_02_pdf_citation
Revises: 20260522_01_audit_log
Create Date: 2026-05-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260522_02_pdf_citation"
down_revision = "20260522_01_audit_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # kb_chunks and kb_documents are created lazily by KnowledgeBaseStore on
    # first use (see app/services/kb_store.py:_init_schema). When alembic
    # runs against a fresh DB they may not exist yet. We use the
    # SQLite-friendly idempotent guard via op.execute with IF NOT EXISTS for
    # the index, and rely on store._init_schema for the base tables. To
    # cover both flows, we create tables here only if they don't already
    # exist, then add columns conditionally.
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "kb_documents" not in existing_tables:
        op.create_table(
            "kb_documents",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("title", sa.Text, nullable=False),
            sa.Column("text", sa.Text, nullable=False),
            sa.Column("created_at", sa.Text, nullable=False),
            sa.Column("source", sa.Text, nullable=False, server_default="text"),
            sa.Column("filename", sa.Text, nullable=True),
            sa.Column("mime_type", sa.Text, nullable=True),
        )

    if "kb_chunks" not in existing_tables:
        op.create_table(
            "kb_chunks",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("document_id", sa.Integer, sa.ForeignKey("kb_documents.id", ondelete="CASCADE"), nullable=False),
            sa.Column("chunk_index", sa.Integer, nullable=False),
            sa.Column("text", sa.Text, nullable=False),
            sa.Column("embedding", sa.LargeBinary, nullable=False),
            sa.Column("embedder", sa.Text, nullable=False, server_default="hash"),
            sa.Column("dim", sa.Integer, nullable=False, server_default="256"),
        )

    # Refresh inspector after potential table creations
    inspector = sa.inspect(bind)

    chunk_cols = {col["name"] for col in inspector.get_columns("kb_chunks")}
    if "page_number" not in chunk_cols:
        op.add_column("kb_chunks", sa.Column("page_number", sa.Integer, nullable=True))

    doc_cols = {col["name"] for col in inspector.get_columns("kb_documents")}
    if "has_original_file" not in doc_cols:
        op.add_column(
            "kb_documents",
            sa.Column("has_original_file", sa.Boolean, nullable=False, server_default=sa.text("0")),
        )
    if "file_relpath" not in doc_cols:
        op.add_column("kb_documents", sa.Column("file_relpath", sa.String(512), nullable=True))

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("kb_chunks")}
    if "idx_kb_chunks_doc_page" not in existing_indexes:
        op.create_index("idx_kb_chunks_doc_page", "kb_chunks", ["document_id", "page_number"])


def downgrade() -> None:
    op.drop_index("idx_kb_chunks_doc_page", table_name="kb_chunks")
    op.drop_column("kb_documents", "file_relpath")
    op.drop_column("kb_documents", "has_original_file")
    op.drop_column("kb_chunks", "page_number")
