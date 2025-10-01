"""Authentication helpers and FastAPI dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable
from uuid import uuid4

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session, select

from app.core.config import get_settings
from app.core.deps import get_ingest_session, get_tenant
from app.models.tenant import TenantRecord
from app.models.user import UserRecord, UserRole
from app.security import InvalidTokenError, create_access_token, decode_token


bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class TokenPair:
    """Container representing issued access and refresh tokens."""

    access_token: str
    refresh_token: str
    expires_in: int


class TokenRegistry:
    """Simple in-memory store for revoked refresh tokens."""

    def __init__(self) -> None:
        self._revoked: set[str] = set()

    def revoke(self, token_id: str | None) -> None:
        if token_id:
            self._revoked.add(token_id)

    def is_revoked(self, token_id: str | None) -> bool:
        return bool(token_id) and token_id in self._revoked


def _get_registry(request: Request) -> TokenRegistry:
    registry = getattr(request.app.state, "token_registry", None)
    if not isinstance(registry, TokenRegistry):
        registry = TokenRegistry()
        request.app.state.token_registry = registry
    return registry


def issue_tokens(
    user: UserRecord,
    *,
    registry: TokenRegistry,
    refresh_ttl_minutes: int = 60 * 24 * 7,
) -> TokenPair:
    """Generate JWT access and refresh tokens for the provided user."""

    claims = {
        "sub": str(user.id),
        "tenant": user.tenant_slug,
        "role": user.role.value,
    }
    access_id = str(uuid4())
    refresh_id = str(uuid4())

    access_payload = {**claims, "type": "access", "jti": access_id}
    refresh_payload = {**claims, "type": "refresh", "jti": refresh_id}

    access_token = create_access_token(access_payload)
    refresh_token = create_access_token(
        refresh_payload,
        expires_delta=timedelta(minutes=refresh_ttl_minutes),
    )
    settings = get_settings()
    expires_in = int(timedelta(minutes=settings.access_token_expire_minutes).total_seconds())
    return TokenPair(access_token=access_token, refresh_token=refresh_token, expires_in=expires_in)


def decode_refresh_token(token: str, *, registry: TokenRegistry) -> dict:
    try:
        payload = decode_token(token)
    except InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_REFRESH_TOKEN") from exc
    if payload.get("type") != "refresh":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_REFRESH_TOKEN")
    token_id = payload.get("jti")
    if token_id and registry.is_revoked(token_id):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="REFRESH_TOKEN_REVOKED")
    return payload


def _load_user(session: Session, user_id: int) -> UserRecord:
    statement = select(UserRecord).where(UserRecord.id == user_id)
    result = session.exec(statement).first()
    if result is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="USER_NOT_FOUND")
    return result


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: Session = Depends(get_ingest_session),
) -> UserRecord:
    """Resolve the current user from the Authorization header."""

    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="NOT_AUTHENTICATED")
    token = credentials.credentials
    try:
        payload = decode_token(token)
    except InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_ACCESS_TOKEN") from exc
    if payload.get("type") != "access":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_ACCESS_TOKEN")
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_ACCESS_TOKEN")
    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_ACCESS_TOKEN") from exc

    registry = _get_registry(request)
    token_id = payload.get("jti")
    if registry.is_revoked(token_id):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="ACCESS_TOKEN_REVOKED")

    user = _load_user(session, user_id_int)
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="USER_INACTIVE")

    tenant = session.exec(select(TenantRecord).where(TenantRecord.slug == user.tenant_slug)).first()
    if tenant is None or not tenant.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="TENANT_DISABLED")

    return user


def get_current_active_user(user: UserRecord = Depends(get_current_user)) -> UserRecord:
    """Ensure the currently authenticated user is active."""

    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="USER_INACTIVE")
    return user


def require_roles(*roles: Iterable[UserRole]):
    """Factory producing a dependency that enforces at least one matching role."""

    role_set = {role for role in roles if isinstance(role, UserRole)}

    def _checker(user: UserRecord = Depends(get_current_active_user)) -> UserRecord:
        if not role_set or user.role in role_set:
            return user
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="INSUFFICIENT_ROLE")

    return _checker


def ensure_tenant_access(
    tenant: str = Depends(get_tenant),
    user: UserRecord = Depends(get_current_active_user),
) -> str:
    """Validate that the authenticated user can operate on the requested tenant."""

    if user.role == UserRole.ADMIN:
        return tenant
    if tenant != user.tenant_slug:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="TENANT_ACCESS_DENIED")
    return tenant


def get_token_registry(request: Request) -> TokenRegistry:
    """Expose the token registry as a dependency."""

    return _get_registry(request)


require_admin_user = require_roles(UserRole.ADMIN)


__all__ = [
    "TokenPair",
    "TokenRegistry",
    "decode_refresh_token",
    "ensure_tenant_access",
    "get_current_active_user",
    "get_current_user",
    "get_token_registry",
    "issue_tokens",
    "require_roles",
    "require_admin_user",
]

