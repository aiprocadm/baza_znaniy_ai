"""Tests for `kb-cli reindex` command."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.cli.reindex import reindex_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def populated_db(tmp_path: Path):
    """Create a SQLite KB with 3 documents, several chunks via the real store."""
    from app.services.kb_store import KnowledgeBaseStore

    db_path = tmp_path / "kb_mvp.sqlite"
    store = KnowledgeBaseStore(db_path)
    store.add_document("doc1", text="alpha " * 100)
    store.add_document("doc2", text="beta " * 100)
    store.add_document("doc3", text="gamma " * 100)
    yield db_path


def _count_chunks(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()[0]
    finally:
        conn.close()


def _embedder_distribution(db: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT embedder, COUNT(*) FROM kb_chunks GROUP BY embedder"
        ).fetchall()
        return {row[0]: row[1] for row in rows}
    finally:
        conn.close()


def test_reindex_dry_run_does_not_modify_db(runner: CliRunner, populated_db: Path):
    """--dry-run prints plan without changing rows."""
    before_count = _count_chunks(populated_db)
    before_dist = _embedder_distribution(populated_db)

    result = runner.invoke(
        reindex_app,
        ["--db-path", str(populated_db), "--embedder", "hash", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output

    assert _count_chunks(populated_db) == before_count
    assert _embedder_distribution(populated_db) == before_dist


def test_reindex_swaps_embedder(runner: CliRunner, populated_db: Path):
    """After reindex, all chunks should report the new embedder name."""
    # The "hash" embedder is deterministic and dependency-free → safe to reindex onto itself.
    result = runner.invoke(
        reindex_app,
        ["--db-path", str(populated_db), "--embedder", "hash", "--force-yes"],
    )
    assert result.exit_code == 0, result.output

    dist = _embedder_distribution(populated_db)
    # All chunks should now be hash:<dim>
    assert all(e.startswith("hash") for e in dist.keys()), dist


def test_reindex_atomic_rollback_on_failure(
    runner: CliRunner, populated_db: Path, monkeypatch
):
    """If embedding fails midway, old kb_chunks survive unchanged."""
    before_count = _count_chunks(populated_db)

    def broken_embedder(*_args, **_kwargs):
        raise RuntimeError("synthetic embed failure")

    # Patch the embedder factory used by reindex
    monkeypatch.setattr(
        "scripts.cli.reindex._make_embedder",
        lambda name: broken_embedder,
    )

    result = runner.invoke(
        reindex_app,
        ["--db-path", str(populated_db), "--embedder", "hash", "--force-yes"],
    )
    assert result.exit_code != 0
    assert _count_chunks(populated_db) == before_count


def test_reindex_resume_skips_done_documents(
    runner: CliRunner, populated_db: Path
):
    """--from-document-id N processes only docs with id >= N."""
    result = runner.invoke(
        reindex_app,
        [
            "--db-path", str(populated_db),
            "--embedder", "hash",
            "--from-document-id", "2",
            "--force-yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "processed 2 document" in result.output.lower() or "doc 2" in result.output.lower()
