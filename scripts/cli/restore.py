"""kb-cli restore — extract a kb-cli backup archive into a data directory."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tarfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import typer

restore_app = typer.Typer(
    name="restore",
    help="Restore a kb-cli backup archive into a data directory.",
    add_completion=False,
)


class RestoreMode(str, Enum):
    replace = "replace"
    merge = "merge"


def _detect_target_embedder(db_path: Path) -> str:
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


def _is_non_empty(data_dir: Path) -> bool:
    if (data_dir / "kb_mvp.sqlite").is_file():
        return True
    kb_files = data_dir / "kb_files"
    if kb_files.is_dir() and any(kb_files.glob("*.pdf")):
        return True
    return False


@restore_app.callback(invoke_without_command=True)
def restore(
    archive: Path = typer.Argument(..., help="Path to the .tar.gz produced by `kb-cli backup`."),
    data_dir: Path = typer.Option(
        Path("var/data"),
        "--data-dir",
        help="Target data directory (will receive kb_mvp.sqlite + kb_files/).",
    ),
    mode: RestoreMode = typer.Option(
        RestoreMode.replace,
        "--mode",
        help="replace: overwrite target; merge: add files but do not overwrite existing.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the interactive confirmation prompt."
    ),
) -> None:
    """Restore a backup archive into a data directory."""

    if not archive.is_file():
        typer.echo(f"ERROR: archive not found: {archive}", err=True)
        raise typer.Exit(code=2)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest from archive (without extracting yet)
    with tarfile.open(archive, "r:gz") as tf:
        manifest_member = tf.extractfile("manifest.json")
        if manifest_member is None:
            typer.echo("ERROR: archive missing manifest.json", err=True)
            raise typer.Exit(code=2)
        manifest = json.loads(manifest_member.read())

    if manifest.get("version") != "1.0":
        typer.echo(
            f"WARNING: unknown manifest version {manifest.get('version')!r} — "
            "proceeding anyway.",
            err=True,
        )

    src_embedder = manifest.get("embedder_used", "unknown")
    tgt_embedder = _detect_target_embedder(data_dir / "kb_mvp.sqlite")
    if src_embedder != "unknown" and tgt_embedder != "unknown" and src_embedder != tgt_embedder:
        typer.echo(
            f"WARNING: embedder mismatch — archive was created with {src_embedder!r}, "
            f"target has {tgt_embedder!r}. Search may need reindex after restore."
        )

    non_empty = _is_non_empty(data_dir)
    if non_empty and mode is RestoreMode.replace:
        if not yes:
            confirm = typer.confirm(
                f"Target {data_dir} is non-empty. Replace contents?",
                default=False,
            )
            if not confirm:
                typer.echo("Aborted by user.")
                raise typer.Exit(code=1)

        # Move existing kb_files/ to a timestamped sibling for safety
        existing_blobs = data_dir / "kb_files"
        if existing_blobs.is_dir() and any(existing_blobs.iterdir()):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            sibling = data_dir / f"kb_files.bak-{stamp}"
            shutil.move(str(existing_blobs), str(sibling))
            typer.echo(f"Backed up existing kb_files/ → {sibling.name}")

    # Extract members. For merge mode, skip files that already exist.
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name == "manifest.json":
                continue
            target_path = data_dir / member.name
            if mode is RestoreMode.merge and target_path.exists():
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:  # directory entry
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.write_bytes(extracted.read())

    typer.echo(f"OK: restored {archive.name} into {data_dir} (mode={mode.value})")
