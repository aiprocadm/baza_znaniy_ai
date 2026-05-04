"""Administrative endpoints for monitoring background jobs."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from app.core.auth import ensure_tenant_access, require_admin_user
from app.core.deps import get_ingest_session
from app.models import JobInfo, JobsResponse
from app.models.file import JobRecord

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/jobs", response_model=JobsResponse)
def list_jobs(
    *,
    session: Session = Depends(get_ingest_session),
    tenant_slug: Optional[str] = Query(None, alias="tenant"),
    job_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    _: str = Depends(ensure_tenant_access),
    __= Depends(require_admin_user),
) -> JobsResponse:
    """Return information about queued and historical jobs."""

    statement = select(JobRecord).order_by(JobRecord.created_at.desc())
    if tenant_slug:
        statement = statement.where(JobRecord.tenant_slug == tenant_slug)
    if job_type:
        statement = statement.where(JobRecord.job_type == job_type)
    if status:
        statement = statement.where(JobRecord.status == status)

    records = session.exec(statement).all()
    items = [
        JobInfo(
            id=str(record.id),
            tenant_slug=record.tenant_slug,
            job_type=record.job_type,
            status=record.status,
            priority=record.priority,
            error=record.error,
            attempt=record.attempt,
            resource_id=record.resource_id,
            payload=record.payload,
            created_at=record.created_at,
            updated_at=record.updated_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
        )
        for record in records
    ]
    return JobsResponse(jobs=items)


__all__ = ["router"]
