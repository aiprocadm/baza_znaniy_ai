from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BillingChargeResult:
    provider_event_id: str
    status: str


class BillingProvider:
    def create_subscription(self, tenant_id: int, plan_code: str) -> str:
        raise NotImplementedError

    def charge_usage(self, tenant_id: int, amount_cents: int, currency: str = "USD") -> BillingChargeResult:
        raise NotImplementedError


class MockBillingProvider(BillingProvider):
    def create_subscription(self, tenant_id: int, plan_code: str) -> str:
        return f"mock-sub-{tenant_id}-{plan_code}"

    def charge_usage(self, tenant_id: int, amount_cents: int, currency: str = "USD") -> BillingChargeResult:
        return BillingChargeResult(provider_event_id=f"mock-charge-{tenant_id}-{amount_cents}", status="processed")


billing_provider: BillingProvider = MockBillingProvider()
