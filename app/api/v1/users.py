"""User management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.core.auth import require_admin_user
from app.core.audit import log_security_event
from app.core.datetime_utils import utc_now
from app.core.deps import get_ingest_session
from app.models.tenant import TenantRecord
from app.models.user import UserCreate, UserRecord, UserResponse, UserRole
from app.security import hash_password

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserResponse], dependencies=[Depends(require_admin_user)])
def list_users(session: Session = Depends(get_ingest_session)) -> list[UserResponse]:
    """List all registered users."""

    users = session.exec(select(UserRecord)).all()
    return [
        UserResponse(
            id=user.id or 0,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            tenant_slug=user.tenant_slug,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login_at=user.last_login_at,
        )
        for user in users
    ]


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin_user)],
)
def create_user(
    payload: UserCreate,
    session: Session = Depends(get_ingest_session),
) -> UserResponse:
    """Create a new user for a tenant."""

    if session.exec(select(UserRecord).where(UserRecord.email == payload.email)).first():
        raise HTTPException(status.HTTP_409_CONFLICT, detail="USER_EXISTS")

    tenant = session.exec(
        select(TenantRecord).where(TenantRecord.slug == payload.tenant_slug)
    ).first()
    if tenant is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="TENANT_NOT_FOUND")

    now = utc_now()
    record = UserRecord(
        email=payload.email,
        full_name=payload.full_name,
        role=payload.role,
        is_active=payload.is_active,
        tenant_slug=payload.tenant_slug,
        hashed_password=hash_password(payload.password),
        created_at=now,
        updated_at=now,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    if str(record.role) != UserRole.MEMBER.value:
        log_security_event("role_change", user_id=record.id, new_role=str(record.role))

    return UserResponse(
        id=record.id or 0,
        email=record.email,
        full_name=record.full_name,
        role=record.role,
        is_active=record.is_active,
        tenant_slug=record.tenant_slug,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_login_at=record.last_login_at,
    )
