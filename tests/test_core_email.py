"""Tests for the email normalisation utilities."""

from __future__ import annotations

import logging

import pytest

from app.core.email import EmailValidationError, normalise_email


def test_normalise_email_success() -> None:
    """The helper should defer to ``email_validator`` when available."""

    assert normalise_email("User@Example.COM") == "User@example.com"


def test_normalise_email_fallback_when_idna_missing(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """When ``email_validator`` raises ``AttributeError`` we use the fallback validator."""

    def _broken_validator(email: str, check_deliverability: bool = False):  # type: ignore[override]
        raise AttributeError("module 'idna' has no attribute 'uts46_remap'")

    monkeypatch.setattr("app.core.email.validate_email", _broken_validator)

    with caplog.at_level(logging.WARNING):
        assert normalise_email("fallback@example.com") == "fallback@example.com"

    assert any(
        "email-validator missing IDNA support" in record.message for record in caplog.records
    ), "expected fallback warning to be logged"


def test_normalise_email_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid email addresses propagate as :class:`EmailValidationError`."""

    from email_validator import EmailNotValidError

    def _reject(email: str, check_deliverability: bool = False):  # type: ignore[override]
        raise EmailNotValidError("bad email")

    monkeypatch.setattr("app.core.email.validate_email", _reject)

    with pytest.raises(EmailValidationError):
        normalise_email("not-an-email")
