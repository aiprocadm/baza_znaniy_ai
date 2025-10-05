"""Regression tests for SQLite engine configuration in ``app.models.file``."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.models import file as file_module


def test_get_engine_applies_sqlite_pragmas(tmp_path, monkeypatch) -> None:
    """Ensure SQLite engines created via ``get_engine`` apply expected PRAGMAs."""

    file_module.get_engine.cache_clear()

    executed_statements: list[str] = []

    real_connect = file_module.sqlite3.connect

    def _tracking_connect(*args: Any, **kwargs: Any):
        connection = real_connect(*args, **kwargs)
        connection.set_trace_callback(lambda sql: executed_statements.append(sql.strip()))
        return connection

    monkeypatch.setattr(file_module.sqlite3, "connect", _tracking_connect)

    db_path = Path(tmp_path) / "pragma-settings.sqlite"
    engine = file_module.get_engine(f"sqlite:///{db_path}", create_schema=False)
    try:
        with sqlite3.connect(db_path) as sqlite_conn:
            journal_mode_row = sqlite_conn.execute("PRAGMA journal_mode").fetchone()
            busy_timeout_row = sqlite_conn.execute("PRAGMA busy_timeout").fetchone()

        assert journal_mode_row is not None
        assert str(journal_mode_row[0]).lower() == "wal"
        assert busy_timeout_row is not None
        assert int(busy_timeout_row[0]) == 5000
    finally:
        engine.dispose()
        file_module.get_engine.cache_clear()

    if executed_statements:
        assert "PRAGMA journal_mode=WAL" in executed_statements
        assert "PRAGMA synchronous=NORMAL" in executed_statements
        assert "PRAGMA busy_timeout=5000" in executed_statements
