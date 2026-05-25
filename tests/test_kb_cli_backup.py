"""Tests for `kb-cli backup` command."""

from __future__ import annotations

import json
import sqlite3
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.cli.backup import backup_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    """Create a fake KB layout: var/data/kb_mvp.sqlite + var/data/kb_files/."""
    data_dir = tmp_path / "var" / "data"
    data_dir.mkdir(parents=True)

    # Create a non-empty SQLite to back up
    db_path = data_dir / "kb_mvp.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE meta(k TEXT, v TEXT)")
    conn.execute("INSERT INTO meta VALUES ('embedder', 'hash')")
    conn.commit()
    conn.close()

    # A blob file
    kb_files = data_dir / "kb_files"
    kb_files.mkdir()
    (kb_files / "1.pdf").write_bytes(b"%PDF-1.4\nhello\n")

    return tmp_path


def test_backup_creates_targz_with_expected_contents(runner: CliRunner, kb_dir: Path) -> None:
    """tar.gz contains kb_mvp.sqlite, kb_files/, manifest.json."""
    out_archive = kb_dir / "out.tar.gz"
    result = runner.invoke(
        backup_app,
        ["--data-dir", str(kb_dir / "var" / "data"), str(out_archive)],
    )
    assert result.exit_code == 0, result.output
    assert out_archive.exists()

    with tarfile.open(out_archive, "r:gz") as tf:
        names = {m.name for m in tf.getmembers()}
    assert "kb_mvp.sqlite" in names
    assert "kb_files/1.pdf" in names
    assert "manifest.json" in names


def test_backup_manifest_has_required_fields(runner: CliRunner, kb_dir: Path) -> None:
    out_archive = kb_dir / "out.tar.gz"
    result = runner.invoke(
        backup_app,
        ["--data-dir", str(kb_dir / "var" / "data"), str(out_archive)],
    )
    assert result.exit_code == 0, result.output

    with tarfile.open(out_archive, "r:gz") as tf:
        manifest_member = tf.extractfile("manifest.json")
        assert manifest_member is not None
        manifest = json.loads(manifest_member.read())

    assert manifest["version"] == "1.0"
    assert "created_at" in manifest
    assert manifest["kb_mvp_db_path"].endswith("kb_mvp.sqlite")
    assert manifest["file_count"] == 1
    assert manifest["total_bytes"] > 0
    assert manifest["embedder_used"] == "hash"


def test_backup_handles_empty_kb_files_dir(runner: CliRunner, tmp_path: Path) -> None:
    """If kb_files/ is empty or missing, manifest.file_count == 0."""
    data_dir = tmp_path / "var" / "data"
    data_dir.mkdir(parents=True)
    db_path = data_dir / "kb_mvp.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE meta(k TEXT, v TEXT)")
    conn.commit()
    conn.close()

    out = tmp_path / "out.tar.gz"
    result = runner.invoke(
        backup_app,
        ["--data-dir", str(data_dir), str(out)],
    )
    assert result.exit_code == 0, result.output

    with tarfile.open(out, "r:gz") as tf:
        manifest = json.loads(tf.extractfile("manifest.json").read())
    assert manifest["file_count"] == 0


def test_backup_fails_if_db_missing(runner: CliRunner, tmp_path: Path) -> None:
    """Missing kb_mvp.sqlite → non-zero exit with helpful message."""
    data_dir = tmp_path / "var" / "data"
    data_dir.mkdir(parents=True)

    out = tmp_path / "out.tar.gz"
    result = runner.invoke(
        backup_app,
        ["--data-dir", str(data_dir), str(out)],
    )
    assert result.exit_code != 0
    assert "kb_mvp.sqlite" in result.output.lower()
