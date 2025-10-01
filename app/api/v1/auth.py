"""Authentication endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from sqlmodel import Session, select

from app.core.auth import TokenPair, bearer_scheme, decode_refresh_token, get_token_registry, issue_tokens
from app.core.deps import get_ingest_session
from app.models.tenant import TenantRecord
from app.models.user import UserRecord
from app.security import InvalidTokenError, decode_token, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = Field(default="bearer")
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class LogoutRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


def _make_token_response(tokens: TokenPair) -> TokenResponse:
    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        expires_in=tokens.expires_in,
    )


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    session: Session = Depends(get_ingest_session),
    registry=Depends(get_token_registry),
) -> TokenResponse:
    """Authenticate the user and return JWT tokens."""

    statement = select(UserRecord).where(UserRecord.email == payload.email)
    user = session.exec(statement).first()
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_CREDENTIALS")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="USER_INACTIVE")
    tenant = session.exec(select(TenantRecord).where(TenantRecord.slug == user.tenant_slug)).first()
    if tenant is None or not tenant.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="TENANT_DISABLED")

    user.last_login_at = datetime.utcnow()
    user.updated_at = datetime.utcnow()
    session.add(user)
    session.commit()
    session.refresh(user)

    tokens = issue_tokens(user, registry=registry)
    return _make_token_response(tokens)


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(
    payload: RefreshRequest,
    session: Session = Depends(get_ingest_session),
    registry=Depends(get_token_registry),
) -> TokenResponse:
    """Exchange a refresh token for a new pair of tokens."""

    refresh_payload = decode_refresh_token(payload.refresh_token, registry=registry)
    user_id = refresh_payload.get("sub")
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_REFRESH_TOKEN")
    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_REFRESH_TOKEN") from exc

    user = session.exec(select(UserRecord).where(UserRecord.id == user_id_int)).first()
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="USER_INACTIVE")

    tenant = session.exec(select(TenantRecord).where(TenantRecord.slug == user.tenant_slug)).first()
    if tenant is None or not tenant.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="TENANT_DISABLED")

    registry.revoke(refresh_payload.get("jti"))
    tokens = issue_tokens(user, registry=registry)
    return _make_token_response(tokens)


@router.post("/logout", response_model=dict)
def logout(
    payload: LogoutRequest,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    registry=Depends(get_token_registry),
) -> dict:
    """Invalidate refresh (and optionally access) tokens."""

    registry.revoke(decode_refresh_token(payload.refresh_token, registry=registry).get("jti"))

    if credentials is not None:
        try:
            access_payload = decode_token(credentials.credentials)
        except InvalidTokenError:
            access_payload = None
        if access_payload is not None:
            registry.revoke(access_payload.get("jti"))

    return {"ok": True}

