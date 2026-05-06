from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any


@dataclass
class UsageEvent:
    tenant_id: str
    subject_type: str
    subject_id: str
    event_type: str
    payload: dict[str, Any]
    idempotency_key: str | None = None


@dataclass
class BillingEvent:
    tenant_id: str
    subject_type: str
    subject_id: str
    event_type: str
    amount: float
    currency: str = "USD"
    payload: dict[str, Any] | None = None
    idempotency_key: str | None = None


class UsageSink(Protocol):
    def write(self, event: UsageEvent) -> None: ...


class BillingSink(Protocol):
    def write(self, event: BillingEvent) -> None: ...


class NoopUsageSink:
    def write(self, event: UsageEvent) -> None:
        return None


class NoopBillingSink:
    def write(self, event: BillingEvent) -> None:
        return None
