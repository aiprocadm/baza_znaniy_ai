"""Tests for `kb-cli restore` command."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.cli.backup import backup_app
from scripts.cli.restore import restore_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_kb(tmp_path: Path, *, embedder: str = "hash") -> Path:
    data_dir = tmp_path / "var" / "data"
    data_dir.mkdir(parents=True)
    db = data_dir / "kb_mvp.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE meta(k TEXT, v TEXT)")
    conn.execute("INSERT INTO meta VALUES ('embedder', ?)", (embedder,))
    conn.commit()
    conn.close()
    (data_dir / "kb_files").mkdir()
    (data_dir / "kb_files" / "1.pdf").write_bytes(b"%PDF-1.4\nA\n")
    return data_dir


def _make_archive(runner: CliRunner, src_data_dir: Path, archive: Path) -> None:
    result = runner.invoke(
        backup_app,
        ["--data-dir", str(src_data_dir), str(archive)],
    )
    assert result.exit_code == 0, result.output


def test_restore_into_empty_target(runner: CliRunner, tmp_path: Path) -> None:
    """Fresh target dir → file extracted, no prompt needed."""
    src = _make_kb(tmp_path / "src")
    archive = tmp_path / "out.tar.gz"
    _make_archive(runner, src, archive)

    target = tmp_path / "target" / "var" / "data"
    target.mkdir(parents=True)
    result = runner.invoke(
        restore_app,
        ["--data-dir", str(target), "--yes", str(archive)],
    )
    assert result.exit_code == 0, result.output
    assert (target / "kb_mvp.sqlite").is_file()
    assert (target / "kb_files" / "1.pdf").is_file()


def test_restore_replace_requires_yes_when_non_empty(runner: CliRunner, tmp_path: Path) -> None:
    """Non-empty target without --yes → aborts."""
    src = _make_kb(tmp_path / "src")
    archive = tmp_path / "out.tar.gz"
    _make_archive(runner, src, archive)

    target = _make_kb(tmp_path / "target")
    result = runner.invoke(
        restore_app,
        ["--data-dir", str(target), str(archive)],
        input="n\n",
    )
    assert result.exit_code != 0


def test_restore_replace_with_yes_creates_backup_of_blobs(
    runner: CliRunner, tmp_path: Path
) -> None:
    """`--yes --mode replace` against non-empty: existing kb_files moved to *.bak-*."""
    src = _make_kb(tmp_path / "src")
    archive = tmp_path / "out.tar.gz"
    _make_archive(runner, src, archive)

    target = _make_kb(tmp_path / "target")
    # Add a unique file in target's kb_files so we can detect the backup
    (target / "kb_files" / "old.pdf").write_bytes(b"old")

    result = runner.invoke(
        restore_app,
        ["--data-dir", str(target), "--yes", "--mode", "replace", str(archive)],
    )
    assert result.exit_code == 0, result.output

    # New files restored
    assert (target / "kb_files" / "1.pdf").is_file()
    # Old kb_files preserved as a sibling directory like kb_files.bak-YYYYMMDD-HHMMSS
    siblings = list(target.glob("kb_files.bak-*"))
    assert siblings, "expected a kb_files.bak-* backup directory"
    assert (siblings[0] / "old.pdf").is_file()


def test_restore_merge_keeps_existing_files(runner: CliRunner, tmp_path: Path) -> None:
    """--mode merge does not overwrite existing files; only adds new ones."""
    src = _make_kb(tmp_path / "src")
    archive = tmp_path / "out.tar.gz"
    _make_archive(runner, src, archive)

    target = _make_kb(tmp_path / "target")
    (target / "kb_files" / "2.pdf").write_bytes(b"target-only")

    result = runner.invoke(
        restore_app,
        ["--data-dir", str(target), "--yes", "--mode", "merge", str(archive)],
    )
    assert result.exit_code == 0, result.output
    # Existing file preserved
    assert (target / "kb_files" / "2.pdf").read_bytes() == b"target-only"
    # Archive file added
    assert (target / "kb_files" / "1.pdf").is_file()


def test_restore_warns_on_embedder_mismatch(runner: CliRunner, tmp_path: Path) -> None:
    """Different embedder in source vs target → warning printed but proceed."""
    src = _make_kb(tmp_path / "src", embedder="ollama:nomic")
    archive = tmp_path / "out.tar.gz"
    _make_archive(runner, src, archive)

    target = _make_kb(tmp_path / "target", embedder="hash")
    result = runner.invoke(
        restore_app,
        ["--data-dir", str(target), "--yes", "--mode", "replace", str(archive)],
    )
    assert result.exit_code == 0
    assert "embedder" in result.output.lower()
