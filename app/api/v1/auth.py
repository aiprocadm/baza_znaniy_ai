"""Authentication endpoints."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session, select

from app.core.audit import log_security_event
from app.core.datetime_utils import utc_now
from app.core.email import EmailValidationError, normalise_email
from app.core.auth import (
    TokenPair,
    _extract_bearer_token,
    decode_refresh_token,
    get_token_registry,
    issue_tokens,
)
from app.core.deps import get_ingest_session
from app.models.tenant import TenantRecord
from app.models.user import UserRecord
from app.security import InvalidTokenError, decode_token, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])
_LOGIN_ATTEMPTS: dict[str, list] = defaultdict(list)
_MAX_ATTEMPTS = 5
_WINDOW = timedelta(minutes=5)


def _client_key(request: Request, email: str) -> str:
    host = request.client.host if request and request.client else "unknown"
    return f"{host}:{email}"


def _check_rate_limit(request: Request, email: str) -> None:
    now = utc_now()
    key = _client_key(request, email)
    attempts = [ts for ts in _LOGIN_ATTEMPTS[key] if now - ts < _WINDOW]
    _LOGIN_ATTEMPTS[key] = attempts
    if len(attempts) >= _MAX_ATTEMPTS:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail="TOO_MANY_LOGIN_ATTEMPTS")


def _record_failed_attempt(request: Request, email: str) -> None:
    _LOGIN_ATTEMPTS[_client_key(request, email)].append(utc_now())


class LoginRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=1)

    @field_validator("email")
    @classmethod
    def _validate_email(cls, value: str) -> str:
        try:
            return normalise_email(value)
        except EmailValidationError as exc:
            raise ValueError("INVALID_EMAIL_FORMAT") from exc


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
    request: Request,
    session: Session = Depends(get_ingest_session),
    registry=Depends(get_token_registry),
) -> TokenResponse:
    """Authenticate the user and return JWT tokens."""
    _check_rate_limit(request, payload.email)
    statement = select(UserRecord).where(UserRecord.email == payload.email)
    user = session.exec(statement).first()
    if user is None or not verify_password(payload.password, user.hashed_password):
        _record_failed_attempt(request, payload.email)
        log_security_event("login_fail", email=payload.email)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_CREDENTIALS")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="USER_INACTIVE")
    tenant = session.exec(select(TenantRecord).where(TenantRecord.slug == user.tenant_slug)).first()
    if tenant is None or not tenant.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="TENANT_DISABLED")

    now = utc_now()
    user.last_login_at = now
    user.updated_at = now
    session.add(user)
    session.commit()
    session.refresh(user)

    log_security_event("login_success", user_id=user.id, tenant=user.tenant_slug)
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
    log_security_event("refresh_rotate", user_id=user.id, revoked_jti=refresh_payload.get("jti"))
    tokens = issue_tokens(user, registry=registry)
    return _make_token_response(tokens)


@router.post("/logout", response_model=dict)
def logout(
    payload: LogoutRequest,
    request: Request,
    registry=Depends(get_token_registry),
) -> dict:
    """Invalidate refresh (and optionally access) tokens."""

    refresh_payload = decode_refresh_token(
        payload.refresh_token, registry=registry, allow_revoked=True
    )
    registry.revoke(refresh_payload.get("jti"))
    registry.mark_inactive(refresh_payload.get("sub"))
    log_security_event("logout", user_id=refresh_payload.get("sub"), revoked_jti=refresh_payload.get("jti"))

    token = _extract_bearer_token(request)

    if token:
        try:
            access_payload = decode_token(token)
        except InvalidTokenError:
            access_payload = None
        if access_payload is not None:
            registry.revoke(access_payload.get("jti"))
            registry.mark_inactive(access_payload.get("sub"))

    return {"ok": True}
