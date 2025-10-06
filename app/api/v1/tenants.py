"""Tenant management endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.core.auth import require_admin_user
from app.core.deps import get_ingest_session
from app.models.tenant import TenantCreate, TenantRecord, TenantResponse

router = APIRouter(prefix="/tenants", tags=["tenants"])


@router.get("", response_model=list[TenantResponse], dependencies=[Depends(require_admin_user)])
def list_tenants(session: Session = Depends(get_ingest_session)) -> list[TenantResponse]:
    """Return all registered tenants."""

    tenants = session.exec(select(TenantRecord)).all()
    return [
        TenantResponse(
            slug=tenant.slug,
            name=tenant.name,
            is_active=tenant.is_active,
            contact_email=tenant.contact_email,
            status=tenant.status,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
        )
        for tenant in tenants
    ]


@router.post("", response_model=TenantResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_admin_user)])
def create_tenant(
    payload: TenantCreate,
    session: Session = Depends(get_ingest_session),
) -> TenantResponse:
    """Create a new tenant entry."""

    existing = session.exec(select(TenantRecord).where(TenantRecord.slug == payload.slug)).first()
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="TENANT_EXISTS")

    now = datetime.utcnow()
    record = TenantRecord(
        slug=payload.slug,
        name=payload.name,
        is_active=payload.is_active,
        contact_email=payload.contact_email,
        created_at=now,
        updated_at=now,
    )
    session.add(record)
    session.commit()
    session.refresh(record)

    return TenantResponse(
        slug=record.slug,
        name=record.name,
        is_active=record.is_active,
        contact_email=record.contact_email,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )

