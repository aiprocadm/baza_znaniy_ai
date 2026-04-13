"""Minimal local fallback for environments without ``email-validator`` installed."""

from __future__ import annotations

from dataclasses import dataclass


class EmailNotValidError(ValueError):
    """Compatibility exception matching the external package API."""


@dataclass(slots=True)
class _ValidatedEmail:
    normalized: str


def validate_email(email: str, check_deliverability: bool = False) -> _ValidatedEmail:
    """Validate and normalise basic email addresses.

    This intentionally implements a tiny subset of the upstream package API
    used in tests and development environments.
    """

    del check_deliverability
    candidate = str(email).strip()
    if "@" not in candidate:
        raise EmailNotValidError("An email address must contain @")
    local, _, domain = candidate.partition("@")
    if not local or not domain:
        raise EmailNotValidError("Invalid email format")
    return _ValidatedEmail(normalized=f"{local}@{domain.lower()}")
