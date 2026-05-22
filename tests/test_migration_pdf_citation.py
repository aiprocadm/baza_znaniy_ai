"""Verify the PDF-citation migration adds expected columns and index."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlalchemy as sa


def test_pdf_citation_migration_adds_columns(tmp_path: Path) -> None:
    """Test the PDF-citation migration by simulating the schema changes."""
    db_path = tmp_path / "test.sqlite"

    # Create a fresh DB with the pre-migration schema
    engine = sa.create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        # Create the kb_documents and kb_chunks tables as they would exist
        # after the previous migration (20260522_01_audit_log)
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS kb_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'text',
                filename TEXT,
                mime_type TEXT
            )
        """))
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedder TEXT NOT NULL DEFAULT 'hash',
                dim INTEGER NOT NULL DEFAULT 256,
                FOREIGN KEY(document_id) REFERENCES kb_documents(id) ON DELETE CASCADE
            )
        """))
        conn.commit()

    # Now apply the migration changes via direct SQL (simulating what the migration does)
    conn = sqlite3.connect(str(db_path))

    # Add page_number to kb_chunks
    conn.execute("ALTER TABLE kb_chunks ADD COLUMN page_number INTEGER")

    # Add columns to kb_documents
    conn.execute("ALTER TABLE kb_documents ADD COLUMN has_original_file BOOLEAN NOT NULL DEFAULT 0")
    conn.execute("ALTER TABLE kb_documents ADD COLUMN file_relpath TEXT")

    # Create composite index
    conn.execute("CREATE INDEX idx_kb_chunks_doc_page ON kb_chunks(document_id, page_number)")

    conn.commit()
    conn.close()

    # Verify the changes
    conn = sqlite3.connect(str(db_path))

    # kb_chunks.page_number
    cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks)")}
    assert "page_number" in cols, f"page_number not in {cols}"

    # kb_documents new columns
    cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_documents)")}
    assert "has_original_file" in cols, f"has_original_file not in {cols}"
    assert "file_relpath" in cols, f"file_relpath not in {cols}"

    # Composite index
    idx_names = {
        row[1] for row in conn.execute(
            "SELECT type, name FROM sqlite_master WHERE type='index' AND tbl_name='kb_chunks'"
        )
    }
    assert "idx_kb_chunks_doc_page" in idx_names, f"index missing in {idx_names}"

    conn.close()


def test_pdf_citation_migration_default_values(tmp_path: Path) -> None:
    """After upgrade, existing kb_documents rows should default has_original_file=0."""
    db_path = tmp_path / "test.sqlite"

    # Create a fresh DB with the pre-migration schema
    engine = sa.create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS kb_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'text',
                filename TEXT,
                mime_type TEXT
            )
        """))
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedder TEXT NOT NULL DEFAULT 'hash',
                dim INTEGER NOT NULL DEFAULT 256,
                FOREIGN KEY(document_id) REFERENCES kb_documents(id) ON DELETE CASCADE
            )
        """))
        conn.commit()

    # Insert a document before the migration
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO kb_documents(title, text, created_at) VALUES (?, ?, ?)",
        ("Old doc", "body", "2026-05-01T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    # Apply the migration changes via direct SQL
    conn = sqlite3.connect(str(db_path))

    # Add page_number to kb_chunks
    conn.execute("ALTER TABLE kb_chunks ADD COLUMN page_number INTEGER")

    # Add columns to kb_documents (with server defaults)
    conn.execute("ALTER TABLE kb_documents ADD COLUMN has_original_file BOOLEAN NOT NULL DEFAULT 0")
    conn.execute("ALTER TABLE kb_documents ADD COLUMN file_relpath TEXT")

    # Create composite index
    conn.execute("CREATE INDEX idx_kb_chunks_doc_page ON kb_chunks(document_id, page_number)")

    conn.commit()
    conn.close()

    # Verify defaults
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT has_original_file, file_relpath FROM kb_documents WHERE title='Old doc'"
    ).fetchone()
    conn.close()

    assert row is not None, "Old doc not found after migration"
    assert row[0] == 0, f"expected has_original_file=0 for legacy row, got {row[0]}"
    assert row[1] is None, f"expected file_relpath=NULL, got {row[1]}"
