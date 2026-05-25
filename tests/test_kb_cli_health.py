"""Tests for `kb-cli health` command."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from scripts.cli.health import health_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_health_ok_path(runner: CliRunner, monkeypatch):
    """If /health returns 200 and status=ok → exit code 0."""

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "status": "ok",
                "kb_stats": {
                    "documents_count": 3,
                    "chunks_count": 9,
                    "db_size_bytes": 12345,
                    "disk_free_bytes": 5_000_000_000,
                    "last_indexed_at": "2026-05-25T10:00:00Z",
                },
                "compliance_mode": None,
                "compliance_implemented": False,
                "llm": {"selected": "deepseek"},
            }

    def fake_get(url, **_):
        return FakeResponse()

    monkeypatch.setattr("scripts.cli.health.httpx.get", fake_get)
    result = runner.invoke(health_app, ["--base-url", "http://127.0.0.1:8000"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "documents=3" in result.output


def test_health_warn_low_disk(runner: CliRunner, monkeypatch):
    """If disk_free_bytes < 100MB → WARN exit code 1."""

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "status": "ok",
                "kb_stats": {
                    "documents_count": 1, "chunks_count": 1,
                    "db_size_bytes": 1, "disk_free_bytes": 50_000_000,
                    "last_indexed_at": None,
                },
                "compliance_mode": None, "compliance_implemented": False,
                "llm": {"selected": None},
            }

    monkeypatch.setattr("scripts.cli.health.httpx.get", lambda *a, **k: FakeResponse())
    result = runner.invoke(health_app, ["--base-url", "http://127.0.0.1:8000"])
    assert result.exit_code == 1
    assert "WARN" in result.output


def test_health_fail_on_http_error(runner: CliRunner, monkeypatch):
    """If /health returns non-2xx or connection fails → FAIL exit code 2."""

    def raises(*_, **__):
        raise ConnectionError("refused")

    monkeypatch.setattr("scripts.cli.health.httpx.get", raises)
    result = runner.invoke(health_app, ["--base-url", "http://127.0.0.1:8000"])
    assert result.exit_code == 2
    assert "FAIL" in result.output
