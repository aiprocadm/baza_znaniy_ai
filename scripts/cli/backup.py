"""kb-cli backup — pack kb_mvp.sqlite + kb_files/ into a tar.gz with manifest."""

from __future__ import annotations

import io
import json
import sqlite3
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import typer

backup_app = typer.Typer(
    name="backup",
    help="Pack the KB SQLite database and original-file blobs into a tar.gz archive.",
    add_completion=False,
)


def _detect_embedder(db_path: Path) -> str:
    """Best-effort detection of the embedder name currently stored in kb_chunks.

    Returns 'unknown' if the column is missing (e.g. very early schema) or
    the table is empty.
    """
    if not db_path.is_file():
        return "unknown"
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks)")}
            if "embedder" in cols:
                row = conn.execute(
                    "SELECT embedder FROM kb_chunks ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    return str(row[0])
            # Some old test fixtures use a meta table — fallback
            cols_meta = {row[1] for row in conn.execute("PRAGMA table_info(meta)")}
            if {"k", "v"}.issubset(cols_meta):
                row = conn.execute("SELECT v FROM meta WHERE k = 'embedder'").fetchone()
                if row:
                    return str(row[0])
        finally:
            conn.close()
    except sqlite3.Error:
        pass
    return "unknown"


@backup_app.callback(invoke_without_command=True)
def backup(
    output: Path = typer.Argument(..., help="Path for the output .tar.gz archive."),
    data_dir: Path = typer.Option(
        Path("var/data"),
        "--data-dir",
        help="Root of KB data (contains kb_mvp.sqlite and kb_files/).",
        show_default=True,
    ),
) -> None:
    """Create a tar.gz archive of the KB state for backup or transfer."""

    db_path = data_dir / "kb_mvp.sqlite"
    if not db_path.is_file():
        typer.echo(
            f"ERROR: kb_mvp.sqlite not found at {db_path}. "
            "Pass --data-dir if your KB lives elsewhere.",
            err=True,
        )
        raise typer.Exit(code=2)

    kb_files = data_dir / "kb_files"
    file_paths = sorted(kb_files.glob("*.pdf")) if kb_files.is_dir() else []
    total_bytes = sum(p.stat().st_size for p in file_paths) + db_path.stat().st_size

    manifest = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kb_mvp_db_path": str(db_path),
        "file_count": len(file_paths),
        "total_bytes": total_bytes,
        "embedder_used": _detect_embedder(db_path),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tf:
        tf.add(db_path, arcname="kb_mvp.sqlite")
        for path in file_paths:
            tf.add(path, arcname=f"kb_files/{path.name}")
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        tf.addfile(info, io.BytesIO(manifest_bytes))

    typer.echo(
        f"OK: wrote {output} ({manifest['file_count']} files, {manifest['total_bytes']} bytes)"
    )
