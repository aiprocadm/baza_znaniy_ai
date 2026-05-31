"""Tests for the AUDIT_LOG_RETENTION_DAYS setting."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Ensure ``get_settings`` observes environment mutations."""

    from app.core import config

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_audit_retention_defaults_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent configuration leaves purging disabled (0 = keep forever)."""

    monkeypatch.delenv("AUDIT_LOG_RETENTION_DAYS", raising=False)

    from app.core.config import get_settings

    assert get_settings().audit_log_retention_days == 0


def test_audit_retention_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit AUDIT_LOG_RETENTION_DAYS is parsed as an int."""

    monkeypatch.setenv("AUDIT_LOG_RETENTION_DAYS", "90")

    from app.core.config import get_settings

    settings = get_settings()
    assert settings.audit_log_retention_days == 90
    assert isinstance(settings.audit_log_retention_days, int)


def test_audit_retention_rejects_negative() -> None:
    """A negative retention is a configuration error, not a silent fallback."""

    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(audit_log_retention_days=-5)
