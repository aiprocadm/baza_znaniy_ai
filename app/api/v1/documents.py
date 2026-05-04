from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlmodel import Session

from app.core.auth import ensure_tenant_access, get_current_active_user
from app.core.deps import get_ingest_session
from app.models import JobInfo
from app.models.file import DocumentRecord, JobRecord
from app.models.user import UserRecord
from app.retriever.qdrant import QdrantVectorStore
from app.services.reindex_service import ReindexService

router = APIRouter(tags=["documents"])


@router.post("/documents/{document_id}/reindex", response_model=JobInfo, status_code=status.HTTP_202_ACCEPTED)
def reindex_document(
    document_id: int,
    dry_run: bool = Query(default=False, description="Preflight reindex without alias switch"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    _: UserRecord = Depends(get_current_active_user),
    session: Session = Depends(get_ingest_session),
    tenant: str = Depends(ensure_tenant_access),
) -> JobInfo:
    record = session.get(DocumentRecord, document_id)
    if record is None or record.tenant_id != tenant:
        raise HTTPException(status_code=404, detail="DOCUMENT_NOT_FOUND")

    service = ReindexService(QdrantVectorStore())
    result = service.reindex_document(
        document_id=str(document_id),
        idempotency_key=idempotency_key,
        dry_run=dry_run,
    )
    job = JobRecord(
        tenant_id=tenant,
        tenant_slug=tenant,
        job_type="reindex",
        status=result.status,
        resource_id=str(document_id),
        payload={
            "copied": result.copied,
            "alias": result.alias,
            "temp_collection": result.temp_collection,
            "source_collection": result.source_collection,
            "idempotency_key": idempotency_key,
            "dry_run": result.dry_run,
        },
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return JobInfo(
        id=str(job.id), tenant_slug=job.tenant_slug, job_type=job.job_type, status=job.status, priority=job.priority,
        error=job.error, attempt=job.attempt, resource_id=job.resource_id, payload=job.payload,
        created_at=job.created_at, updated_at=job.updated_at, started_at=job.started_at, finished_at=job.finished_at,
    )


@router.get("/ingest/jobs/{job_id}", response_model=JobInfo)
def get_ingest_job(
    job_id: int,
    _: UserRecord = Depends(get_current_active_user),
    session: Session = Depends(get_ingest_session),
    tenant: str = Depends(ensure_tenant_access),
) -> JobInfo:
    job = session.get(JobRecord, job_id)
    if job is None or job.tenant_id != tenant:
        raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")
    return JobInfo(
        id=str(job.id), tenant_slug=job.tenant_slug, job_type=job.job_type, status=job.status, priority=job.priority,
        error=job.error, attempt=job.attempt, resource_id=job.resource_id, payload=job.payload,
        created_at=job.created_at, updated_at=job.updated_at, started_at=job.started_at, finished_at=job.finished_at,
    )
