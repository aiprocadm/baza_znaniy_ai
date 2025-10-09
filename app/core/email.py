"""Utilities for validating and normalising email addresses."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from email_validator import EmailNotValidError, validate_email

LOGGER = logging.getLogger(__name__)

# RFC 5322 is intentionally permissive. For the fallback we opt for a
# conservative ASCII-focused pattern that supports the characters users expect
# while remaining easy to audit. The pattern enforces:
#   * one ``@`` separator
#   * no whitespace
#   * label lengths within the DNS specification (1-63 characters)
#   * local-part limited to the typical ``atext`` characters.
_LOCAL_PART_PATTERN = r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
_LABEL_PATTERN = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_SIMPLE_EMAIL_REGEX = re.compile(
    rf"^(?P<local>{_LOCAL_PART_PATTERN})@(?P<domain>{_LABEL_PATTERN}(?:\.{_LABEL_PATTERN})*)$"
)


@dataclass(slots=True)
class EmailValidationResult:
    """Normalised representation of a validated email address."""

    original: str
    normalised: str


class EmailValidationError(ValueError):
    """Raised when an email address cannot be validated."""


def _fallback_normalise(email: str) -> EmailValidationResult:
    """Fallback ASCII-only email validation used when ``email_validator`` breaks.

    The logic intentionally errs on the side of caution. It accepts common ASCII
    email addresses, lowercases the domain component, and rejects values that
    would clearly violate DNS or email syntax rules.
    """

    candidate = email.strip()
    match = _SIMPLE_EMAIL_REGEX.fullmatch(candidate)
    if match is None:
        raise EmailValidationError("INVALID_EMAIL_FORMAT")

    local = match.group("local")
    domain = match.group("domain")

    if ".." in local:
        raise EmailValidationError("INVALID_EMAIL_FORMAT")

    labels = domain.split(".")
    for label in labels:
        if len(label) == 0:
            raise EmailValidationError("INVALID_EMAIL_FORMAT")

    normalised_domain = domain.lower()
    normalised = f"{local}@{normalised_domain}"
    return EmailValidationResult(original=candidate, normalised=normalised)


def normalise_email(email: str) -> str:
    """Validate and normalise an email address.

    ``email_validator`` provides extensive RFC-compliant validation but has a
    hard dependency on the ``idna`` package. Certain minimal environments ship a
    truncated ``idna`` implementation which raises ``AttributeError``. To keep
    authentication functional we fall back to a conservative ASCII validator in
    that situation while logging the downgrade for observability.
    """

    try:
        info = validate_email(email, check_deliverability=False)
    except EmailNotValidError as exc:  # pragma: no cover - exercised in app tests
        raise EmailValidationError("INVALID_EMAIL_FORMAT") from exc
    except AttributeError as exc:
        LOGGER.warning(
            "email-validator missing IDNA support; falling back to simplified validation",
            extra={"dependency": "email_validator", "error": str(exc)},
        )
        return _fallback_normalise(email).normalised
    else:
        return info.normalized
