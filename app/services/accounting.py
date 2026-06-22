from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any
from sqlmodel import select

from app.models.file import BillingEventRecord, RagRunRecord, RagRunSourceRecord, UsageEventRecord


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


class SqlUsageSink:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def write(self, event: UsageEvent) -> None:
        with self._session_factory() as session:
            if event.idempotency_key:
                existing = session.exec(
                    select(UsageEventRecord).where(
                        UsageEventRecord.tenant_id == event.tenant_id,
                        UsageEventRecord.idempotency_key == event.idempotency_key,
                    )
                ).first()
                if existing is not None:
                    return None
            session.add(
                UsageEventRecord(
                    tenant_id=event.tenant_id,
                    subject_type=event.subject_type,
                    subject_id=event.subject_id,
                    event_type=event.event_type,
                    payload=event.payload,
                    idempotency_key=event.idempotency_key,
                )
            )
            session.commit()
        return None

    def write_rag_run(
        self,
        *,
        tenant_id: str,
        subject_type: str,
        subject_id: str,
        query: str,
        sources: list[dict[str, Any]],
    ) -> None:
        with self._session_factory() as session:
            run = RagRunRecord(
                tenant_id=tenant_id, subject_type=subject_type, subject_id=subject_id, query=query
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            assert run.id is not None  # populated by session.refresh above
            for source in sources:
                session.add(
                    RagRunSourceRecord(
                        rag_run_id=run.id,
                        source_file=source.get("file"),
                        source_page=source.get("page"),
                        score=source.get("score"),
                    )
                )
            session.commit()


class SqlBillingSink:
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def write(self, event: BillingEvent) -> None:
        with self._session_factory() as session:
            if event.idempotency_key:
                existing = session.exec(
                    select(BillingEventRecord).where(
                        BillingEventRecord.tenant_id == event.tenant_id,
                        BillingEventRecord.idempotency_key == event.idempotency_key,
                    )
                ).first()
                if existing is not None:
                    return None
            session.add(
                BillingEventRecord(
                    tenant_id=event.tenant_id,
                    subject_type=event.subject_type,
                    subject_id=event.subject_id,
                    event_type=event.event_type,
                    amount=event.amount,
                    currency=event.currency,
                    payload=event.payload,
                    idempotency_key=event.idempotency_key,
                )
            )
            session.commit()
        return None
