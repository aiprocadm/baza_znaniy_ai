"""Authentication helpers and FastAPI dependencies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable
from uuid import uuid4

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer
from sqlmodel import Session, select

from app.core.config import get_settings
from app.core.deps import get_ingest_session, get_tenant
from app.core.datetime_utils import utc_now
from app.models.tenant import TenantRecord
from app.models.user import UserRecord, UserRole
from app.security import InvalidTokenError, create_access_token, decode_token


bearer_scheme = HTTPBearer(auto_error=False)


_AUTH_DISABLED_ENV_KEYS = (
    "AUTH_DISABLED_FOR_TESTS",
    "AUTH_DISABLED",
    "DISABLE_AUTH",
    "AUTH_DISABLE",
    "KB_DISABLE_AUTH",
)
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_auth_disabled() -> bool:
    """Return whether environment variables disable authentication."""

    for key in _AUTH_DISABLED_ENV_KEYS:
        raw_value = os.getenv(key)
        if raw_value and raw_value.strip().lower() in _TRUTHY_ENV_VALUES:
            return True
    return False


def _build_test_admin_user() -> UserRecord:
    """Return a synthetic admin user for environments with auth disabled."""

    now = utc_now()
    return UserRecord(
        id=0,
        tenant_id="test-tenant",
        tenant_slug="test-tenant",
        email="admin@test.local",
        full_name="Test Admin",
        role=UserRole.ADMIN,
        is_active=True,
        status="active",
        hashed_password="",
        created_at=now,
        updated_at=now,
    )


def _extract_bearer_token(request: Any) -> str | None:
    """Return the bearer token from a request-like object, if present."""

    if request is None:
        return None

    header_value: str | None = None
    headers = getattr(request, "headers", None)
    if headers is not None:
        getter = getattr(headers, "get", None)
        if callable(getter):
            header_value = getter("Authorization") or getter("authorization")

    if header_value is None and hasattr(request, "scope"):
        scope = getattr(request, "scope", {}) or {}
        raw_headers = scope.get("headers") or []
        for key, value in raw_headers:
            try:
                key_text = key.decode().lower()
            except Exception:
                continue
            if key_text == "authorization":
                try:
                    header_value = value.decode()
                except Exception:
                    header_value = None
                break

    if not header_value:
        return None

    scheme, _, token = header_value.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


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
        self._inactive_users: set[str] = set()

    def revoke(self, token_id: str | None) -> None:
        if token_id:
            self._revoked.add(token_id)
            _GLOBAL_REVOKED_TOKENS.add(token_id)

    def is_revoked(self, token_id: str | None) -> bool:
        if not token_id:
            return False
        return token_id in self._revoked or token_id in _GLOBAL_REVOKED_TOKENS

    def mark_active(self, user_id: str | None) -> None:
        if user_id:
            self._inactive_users.discard(user_id)

    def mark_inactive(self, user_id: str | None) -> None:
        if user_id:
            self._inactive_users.add(user_id)

    def is_active(self, user_id: str | None) -> bool:
        if not user_id:
            return False
        return user_id not in self._inactive_users


def _get_registry(request: Request) -> TokenRegistry:
    registry = getattr(request.app.state, "token_registry", None)
    if not isinstance(registry, TokenRegistry):
        registry = _GLOBAL_TOKEN_REGISTRY
        request.app.state.token_registry = registry
    return registry


def issue_tokens(
    user: UserRecord,
    *,
    registry: TokenRegistry,
    refresh_ttl_minutes: int = 60 * 24 * 7,
) -> TokenPair:
    """Generate JWT access and refresh tokens for the provided user."""

    role_value = user.role.value if hasattr(user.role, "value") else str(user.role)
    claims = {
        "sub": str(user.id),
        "tenant": user.tenant_slug,
        "role": role_value,
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
    registry.mark_active(str(user.id))
    return TokenPair(access_token=access_token, refresh_token=refresh_token, expires_in=expires_in)


def decode_refresh_token(token: str, *, registry: TokenRegistry, allow_revoked: bool = False) -> dict:
    try:
        payload = decode_token(token)
    except InvalidTokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_REFRESH_TOKEN") from exc
    if payload.get("type") != "refresh":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_REFRESH_TOKEN")
    token_id = payload.get("jti")
    if token_id and registry.is_revoked(token_id) and not allow_revoked:
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
    session: Session = Depends(get_ingest_session),
) -> UserRecord:
    """Resolve the current user from the Authorization header."""

    settings = get_settings()
    if getattr(settings, "auth_disabled", False) or _env_auth_disabled():
        return _build_test_admin_user()

    token = _extract_bearer_token(request)
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="NOT_AUTHENTICATED")
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

    if not registry.is_active(str(user.id)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="NOT_AUTHENTICATED")

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
    role_values = {role.value for role in role_set}

    def _checker(user: UserRecord = Depends(get_current_active_user)) -> UserRecord:
        settings = get_settings()
        if getattr(settings, "auth_disabled", False) or _env_auth_disabled():
            return user

        if not role_set:
            return user

        user_role = user.role
        if isinstance(user_role, UserRole) and user_role in role_set:
            return user

        role_text = user_role.value if isinstance(user_role, UserRole) else str(user_role)
        if role_text in role_values:
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


_GLOBAL_TOKEN_REGISTRY = TokenRegistry()
_GLOBAL_REVOKED_TOKENS: set[str] = set()


def get_token_registry(request: Any = None) -> TokenRegistry:
    """Expose the token registry as a dependency."""

    global _GLOBAL_TOKEN_REGISTRY
    if request is None:
        return _GLOBAL_TOKEN_REGISTRY
    registry = _get_registry(request)
    _GLOBAL_TOKEN_REGISTRY = registry
    return registry


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
    "_extract_bearer_token",
]

