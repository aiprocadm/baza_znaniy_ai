from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models import Chunk, Document, Plan, Subscription, UsageCounter


@dataclass(frozen=True)
class QuotaSnapshot:
    max_storage_bytes: int
    max_documents: int
    max_search_requests: int
    max_llm_requests: int
    period_start: datetime
    period_end: datetime
    storage_used_bytes: int
    documents_used: int
    search_used: int
    llm_used: int


class PolicyEngine:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_active_subscription(self, tenant_id: int) -> Subscription | None:
        return (
            self.db.execute(
                select(Subscription)
                .where(Subscription.tenant_id == tenant_id, Subscription.status == "active")
                .order_by(Subscription.current_period_start.desc())
                .limit(1)
            )
            .scalars()
            .first()
        )

    def get_plan(self, plan_code: str) -> Plan | None:
        return self.db.execute(select(Plan).where(Plan.code == plan_code)).scalars().first()

    def snapshot(self, tenant_id: int) -> QuotaSnapshot:
        subscription = self.get_active_subscription(tenant_id)
        if subscription is None:
            period_start = datetime.now(timezone.utc)
            period_end = period_start + timedelta(days=30)
            default_limits = (50 * 1024 * 1024, 100, 1000, 1000)
        else:
            period_start = subscription.current_period_start
            period_end = subscription.current_period_end or (period_start + timedelta(days=30))
            plan = self.get_plan(subscription.plan_code)
            default_limits = (
                plan.max_storage_bytes if plan else 50 * 1024 * 1024,
                plan.max_documents if plan else 100,
                plan.max_search_requests if plan else 1000,
                plan.max_llm_requests if plan else 1000,
            )

        documents_used = self.db.execute(select(func.count(Document.id)).where(Document.tenant_id == tenant_id)).scalar_one()
        storage_used_bytes = self.db.execute(select(func.coalesce(func.sum(Chunk.tokens), 0)).where(Chunk.tenant_id == tenant_id)).scalar_one()

        counter = self.db.execute(
            select(UsageCounter).where(
                UsageCounter.tenant_id == tenant_id,
                UsageCounter.period_start == period_start,
                UsageCounter.period_end == period_end,
            )
        ).scalars().first()
        search_used = counter.search_requests if counter else 0
        llm_used = counter.llm_requests if counter else 0

        return QuotaSnapshot(*default_limits, period_start, period_end, storage_used_bytes, documents_used, search_used, llm_used)

    def enforce(self, tenant_id: int, operation: str) -> None:
        q = self.snapshot(tenant_id)
        if operation == "upload":
            if q.documents_used >= q.max_documents:
                raise ValueError("Document quota exceeded")
            if q.storage_used_bytes >= q.max_storage_bytes:
                raise ValueError("Storage quota exceeded")
        elif operation == "search" and q.search_used >= q.max_search_requests:
            raise ValueError("Search request quota exceeded")
        elif operation == "llm" and q.llm_used >= q.max_llm_requests:
            raise ValueError("LLM request quota exceeded")

    def increment_counter(self, tenant_id: int, counter_kind: str, delta: int = 1) -> None:
        q = self.snapshot(tenant_id)
        row = self.db.execute(
            select(UsageCounter).where(
                UsageCounter.tenant_id == tenant_id,
                UsageCounter.period_start == q.period_start,
                UsageCounter.period_end == q.period_end,
            )
        ).scalars().first()
        if row is None:
            row = UsageCounter(
                tenant_id=tenant_id,
                period_start=q.period_start,
                period_end=q.period_end,
                storage_bytes=0,
                documents_count=q.documents_used,
                search_requests=0,
                llm_requests=0,
            )
            self.db.add(row)
        if counter_kind == "search":
            row.search_requests += delta
        elif counter_kind == "llm":
            row.llm_requests += delta
        self.db.commit()
