"""Verify the PDF-citation migration adds expected columns and index.

Tests invoke our migration's upgrade() via alembic.command.upgrade() with
the previous revision pre-stamped, so the upstream chain (which contains
a SQLite-incompatible op.drop_constraint in 20260503_01_target_data_model)
is skipped. This ensures the test actually exercises
alembic/versions/20260522_02_pdf_citation.py's upgrade(), not a manual
duplication of the SQL.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command as alembic_cmd
from alembic.config import Config as AlembicConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_config(db_path: Path) -> AlembicConfig:
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _create_pre_migration_tables(db_path: Path) -> None:
    """Create kb_documents and kb_chunks as they'd exist before our migration."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE kb_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'text',
                filename TEXT,
                mime_type TEXT
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE kb_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedder TEXT NOT NULL DEFAULT 'hash',
                dim INTEGER NOT NULL DEFAULT 256
            )
        """
        )
        conn.commit()
    finally:
        conn.close()


def test_pdf_citation_migration_adds_columns(tmp_path: Path) -> None:
    """Our migration's upgrade() adds page_number, has_original_file,
    file_relpath, and idx_kb_chunks_doc_page."""
    db_path = tmp_path / "test.sqlite"

    # Set up pre-migration state: create the tables manually, then stamp
    # the alembic state at the previous revision so our migration is the
    # only one alembic will run.
    _create_pre_migration_tables(db_path)
    cfg = _make_config(db_path)
    alembic_cmd.stamp(cfg, "20260522_01_audit_log")

    # Now run our migration (and only ours, because everything before is
    # stamped). This actually invokes upgrade() from
    # alembic/versions/20260522_02_pdf_citation.py.
    alembic_cmd.upgrade(cfg, "head")

    # Verify schema
    conn = sqlite3.connect(str(db_path))
    try:
        chunk_cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks)")}
        assert "page_number" in chunk_cols, f"page_number not in {chunk_cols}"

        doc_cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_documents)")}
        assert "has_original_file" in doc_cols, f"has_original_file not in {doc_cols}"
        assert "file_relpath" in doc_cols, f"file_relpath not in {doc_cols}"

        idx_names = {
            row[1]
            for row in conn.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE type='index' AND tbl_name='kb_chunks'"
            )
        }
        assert "idx_kb_chunks_doc_page" in idx_names, f"index missing in {idx_names}"
    finally:
        conn.close()


def test_pdf_citation_migration_default_values(tmp_path: Path) -> None:
    """A pre-existing kb_documents row gets has_original_file=0 and
    file_relpath=NULL after our migration runs."""
    db_path = tmp_path / "test.sqlite"

    _create_pre_migration_tables(db_path)

    # Seed a legacy row before the migration
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO kb_documents(title, text, created_at) VALUES (?, ?, ?)",
            ("Old doc", "body", "2026-05-01T00:00:00Z"),
        )
        conn.commit()
    finally:
        conn.close()

    cfg = _make_config(db_path)
    alembic_cmd.stamp(cfg, "20260522_01_audit_log")
    alembic_cmd.upgrade(cfg, "head")

    # Verify defaults applied to the legacy row
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT has_original_file, file_relpath FROM kb_documents " "WHERE title='Old doc'"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None, "Old doc not found after migration"
    assert row[0] == 0, f"expected has_original_file=0, got {row[0]}"
    assert row[1] is None, f"expected file_relpath=NULL, got {row[1]}"
