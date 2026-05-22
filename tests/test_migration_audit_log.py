"""Test that the audit_log migration creates the table with expected schema.

Strategy: the migrations prior to 20260506_01 are PostgreSQL-specific and
cannot run on SQLite. We stamp the DB at the immediate predecessor revision
(20260506_01_api_keys_usage_rag) so that alembic upgrade head only runs the
new audit_log migration. The audit_log table has no FK dependencies, so no
predecessor tables need to exist.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path


PREV_REVISION = "20260506_01_api_keys_usage_rag"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _alembic(args: list[str], db_url: str) -> subprocess.CompletedProcess:
    import os
    env = {**os.environ, "DB_URL": db_url}
    return subprocess.run(
        [sys.executable, "-m", "alembic"] + args,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_audit_log_table_created_by_migration(tmp_path: Path) -> None:
    """Stamp DB at predecessor revision, upgrade to head, verify audit_log schema."""
    db_path = tmp_path / "test.sqlite"
    db_url = f"sqlite:///{db_path}"

    # Create the alembic_version table by stamping the predecessor revision.
    # This tells alembic the DB is already at that revision without running
    # the PostgreSQL-specific earlier migrations.
    stamp = _alembic(["stamp", PREV_REVISION], db_url)
    assert stamp.returncode == 0, f"alembic stamp failed: {stamp.stderr}"

    # Now upgrade head — should only run 20260522_01_audit_log.
    upgrade = _alembic(["upgrade", "head"], db_url)
    assert upgrade.returncode == 0, f"alembic upgrade failed: {upgrade.stderr}"

    # Verify the table and columns exist.
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'"
    )
    assert cursor.fetchone() is not None, "audit_log table missing"

    cursor = conn.execute("PRAGMA table_info(audit_log)")
    columns = {row[1] for row in cursor.fetchall()}
    expected = {
        "id", "timestamp", "event", "user_id", "tenant",
        "ip", "request_path", "request_method", "status_code",
        "payload_json", "correlation_id",
    }
    assert expected.issubset(columns), f"missing columns: {expected - columns}"

    conn.close()
