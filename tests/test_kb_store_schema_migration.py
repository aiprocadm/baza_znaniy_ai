"""Opening a pre-existing MVP DB must upgrade its schema in place.

``KnowledgeBaseStore._init_schema`` builds the schema with
``CREATE TABLE IF NOT EXISTS`` + ``CREATE INDEX IF NOT EXISTS``. That is a
no-op for tables that already exist, so columns appended to the schema after
a DB was first created (``page_number``, ``has_original_file``,
``file_relpath`` — added in f96b342) never land on an older DB. Opening such
a DB used to raise ``sqlite3.OperationalError: no such column: page_number``
the moment ``idx_kb_chunks_doc_page`` was created.

This mirrors what alembic/versions/20260522_02_pdf_citation.py does for the
full-stack path; the MVP path deliberately does not run alembic, so the
store must reconcile the columns itself on open.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.services.kb_store import KnowledgeBaseStore


def _create_legacy_db(db_path: Path) -> None:
    """Create kb_documents/kb_chunks as they existed before f96b342.

    No ``page_number`` on kb_chunks; no ``has_original_file`` /
    ``file_relpath`` on kb_documents; no ``idx_kb_chunks_doc_page``.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE kb_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'text',
                filename TEXT,
                mime_type TEXT
            );
            CREATE TABLE kb_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedder TEXT NOT NULL DEFAULT 'hash',
                dim INTEGER NOT NULL DEFAULT 256
            );
            CREATE INDEX idx_kb_chunks_doc ON kb_chunks(document_id);
            CREATE INDEX idx_kb_chunks_dim ON kb_chunks(dim);
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_opening_legacy_db_upgrades_schema_in_place(tmp_path: Path) -> None:
    """Constructing the store against a legacy DB upgrades columns + index
    instead of raising OperationalError."""
    db_path = tmp_path / "legacy.sqlite"
    _create_legacy_db(db_path)

    # Must not raise sqlite3.OperationalError: no such column: page_number
    KnowledgeBaseStore(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        chunk_cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks)")}
        doc_cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_documents)")}
        idx_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master " "WHERE type='index' AND tbl_name='kb_chunks'"
            )
        }
    finally:
        conn.close()

    assert "page_number" in chunk_cols, f"page_number missing: {chunk_cols}"
    assert "has_original_file" in doc_cols, f"has_original_file missing: {doc_cols}"
    assert "file_relpath" in doc_cols, f"file_relpath missing: {doc_cols}"
    assert "idx_kb_chunks_doc_page" in idx_names, f"index missing: {idx_names}"


def test_opening_legacy_db_preserves_existing_rows(tmp_path: Path) -> None:
    """A legacy kb_documents row survives the in-place upgrade and gets the
    new columns' defaults (has_original_file=0, file_relpath=NULL)."""
    db_path = tmp_path / "legacy.sqlite"
    _create_legacy_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO kb_documents(title, text, created_at) VALUES (?, ?, ?)",
            ("Old doc", "body", "2026-05-01T00:00:00Z"),
        )
        conn.commit()
    finally:
        conn.close()

    KnowledgeBaseStore(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT has_original_file, file_relpath FROM kb_documents WHERE title='Old doc'"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None, "legacy row vanished after upgrade"
    assert row[0] == 0, f"expected has_original_file=0, got {row[0]}"
    assert row[1] is None, f"expected file_relpath=NULL, got {row[1]}"
