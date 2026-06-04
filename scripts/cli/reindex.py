"""kb-cli reindex — re-embed all chunks atomically using a new embedder."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

import typer

reindex_app = typer.Typer(
    name="reindex",
    help="Re-embed every chunk using the given embedder. Atomic (rollback on failure).",
    add_completion=False,
)


def _make_embedder(name: str) -> Callable[[str], tuple[bytes, str, int]]:
    """Return a function that embeds a single chunk text.

    ``hash`` uses the deterministic, dependency-free fallback. Any other name
    (``ollama`` / ``api`` / …) resolves the real embedder from the environment
    via the canonical builder; ``--embedder`` then asserts which backend we
    expect, so a misconfigured env fails loudly instead of silently
    re-embedding with the hashing fallback (near-random vectors).
    """
    import struct

    from app.services import kb_embeddings

    embedder: kb_embeddings.Embedder
    if name.startswith("hash"):
        embedder = kb_embeddings.HashingEmbedder()
    else:
        embedder = kb_embeddings.get_embedder()
        if embedder.name != name:
            raise typer.BadParameter(
                f"--embedder {name!r} but the configured backend resolved to "
                f"{embedder.name!r}. Set KB_EMBEDDINGS_BACKEND={name} "
                f"(+ model/base env) before reindexing."
            )

    def _embed(text: str) -> tuple[bytes, str, int]:
        vec = embedder.embed(text)
        blob = struct.pack(f"{len(vec)}f", *vec)
        return blob, embedder.name, len(vec)

    return _embed


@reindex_app.callback(invoke_without_command=True)
def reindex(
    db_path: Path = typer.Option(
        Path("var/data/kb_mvp.sqlite"),
        "--db-path",
        help="Path to kb_mvp.sqlite.",
    ),
    embedder: str = typer.Option(..., "--embedder", help="Embedder name (e.g. 'hash')."),
    from_document_id: int = typer.Option(
        1, "--from-document-id", help="Resume from document_id >= N."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan without changes."),
    force_yes: bool = typer.Option(False, "--force-yes", help="Skip confirmation."),
) -> None:
    """Re-embed chunks atomically and swap the kb_chunks table."""

    if not db_path.is_file():
        typer.echo(f"ERROR: {db_path} does not exist", err=True)
        raise typer.Exit(code=2)

    conn = sqlite3.connect(str(db_path))
    try:
        chunk_rows = conn.execute(
            """
            SELECT id, document_id, chunk_index, text, page_number
            FROM kb_chunks
            WHERE document_id >= ?
            ORDER BY document_id, chunk_index
            """,
            (from_document_id,),
        ).fetchall()
    finally:
        conn.close()

    docs_seen = sorted({row[1] for row in chunk_rows})
    typer.echo(
        f"Plan: reindex {len(chunk_rows)} chunks across {len(docs_seen)} documents "
        f"with embedder={embedder!r}, starting from doc_id={from_document_id}"
    )

    if dry_run:
        typer.echo("DRY RUN — no changes made.")
        raise typer.Exit(code=0)

    if not force_yes:
        if not typer.confirm("Proceed?", default=False):
            typer.echo("Aborted by user.")
            raise typer.Exit(code=1)

    embed = _make_embedder(embedder)

    # Atomic strategy: build kb_chunks_new in autocommit, then BEGIN/COMMIT
    # for the DELETE+INSERT+DROP swap. We use isolation_level=None so Python
    # sqlite3 does not implicitly start a transaction before our explicit BEGIN.
    conn = sqlite3.connect(str(db_path))
    conn.isolation_level = None
    try:
        # Snapshot column list so we can mirror schema (page_number may or may not exist).
        chunk_cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks)")}
        has_page = "page_number" in chunk_cols

        conn.execute("DROP TABLE IF EXISTS kb_chunks_new")
        create_sql = (
            "CREATE TABLE kb_chunks_new ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "document_id INTEGER NOT NULL, "
            "chunk_index INTEGER NOT NULL, "
            "text TEXT NOT NULL, "
            "embedding BLOB NOT NULL, "
            "embedder TEXT NOT NULL, "
            "dim INTEGER NOT NULL" + (", page_number INTEGER" if has_page else "") + ")"
        )
        conn.execute(create_sql)

        # Wrap the embed-and-insert loop in a single transaction for speed.
        conn.execute("BEGIN")
        try:
            processed_docs: set[int] = set()
            for row in chunk_rows:
                _chunk_id, doc_id, chunk_idx, text, page_no = row
                blob, embedder_name, dim = embed(text)
                if has_page:
                    conn.execute(
                        "INSERT INTO kb_chunks_new(document_id, chunk_index, text, "
                        "embedding, embedder, dim, page_number) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (doc_id, chunk_idx, text, blob, embedder_name, dim, page_no),
                    )
                else:
                    conn.execute(
                        "INSERT INTO kb_chunks_new(document_id, chunk_index, text, "
                        "embedding, embedder, dim) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (doc_id, chunk_idx, text, blob, embedder_name, dim),
                    )
                processed_docs.add(doc_id)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        # Atomic swap of old → new chunks
        conn.execute("BEGIN")
        try:
            conn.execute("DELETE FROM kb_chunks WHERE document_id >= ?", (from_document_id,))
            cols = "document_id, chunk_index, text, embedding, embedder, dim" + (
                ", page_number" if has_page else ""
            )
            conn.execute(f"INSERT INTO kb_chunks({cols}) SELECT {cols} FROM kb_chunks_new")
            conn.execute("DROP TABLE kb_chunks_new")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    except Exception:
        try:
            conn.execute("DROP TABLE IF EXISTS kb_chunks_new")
        except sqlite3.Error:
            pass
        conn.close()
        raise
    finally:
        try:
            conn.close()
        except sqlite3.ProgrammingError:
            pass

    typer.echo(
        f"OK: processed {len(processed_docs)} document(s), "
        f"re-embedded {len(chunk_rows)} chunk(s)"
    )
