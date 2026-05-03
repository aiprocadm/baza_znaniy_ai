"""Security audit logging helpers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("security.audit")


def log_security_event(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info("security_event", extra={"security_event": payload})

