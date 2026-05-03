from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from typing import Iterator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.app.db.session import get_session_factory

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
security = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthIdentity:
    user_id: str
    email: str
    tenant_id: str
    tenant_slug: str
    roles: tuple[Literal["user", "tenant-admin", "platform-admin"], ...]


ADMIN_IDENTITY = AuthIdentity(
    user_id="u_admin",
    email="admin@kb.ai",
    tenant_id="t_platform",
    tenant_slug="platform",
    roles=("platform-admin",),
)
ADMIN_PASSWORD_HASH = pwd_context.hash("secret")

TENANT_ADMIN_IDENTITY = AuthIdentity(
    user_id="u_tenant_admin",
    email="tenant-admin@kb.ai",
    tenant_id="t_alpha",
    tenant_slug="alpha",
    roles=("tenant-admin",),
)
TENANT_USER_IDENTITY = AuthIdentity(
    user_id="u_tenant_user",
    email="tenant-user@kb.ai",
    tenant_id="t_alpha",
    tenant_slug="alpha",
    roles=("user",),
)
OTHER_TENANT_ADMIN_IDENTITY = AuthIdentity(
    user_id="u_tenant_admin_b",
    email="tenant-admin-b@kb.ai",
    tenant_id="t_beta",
    tenant_slug="beta",
    roles=("tenant-admin",),
)

_TOKEN_TO_IDENTITY: dict[str, AuthIdentity] = {
    "kb_admin_token": ADMIN_IDENTITY,
    "kb_tenant_admin_token": TENANT_ADMIN_IDENTITY,
    "kb_tenant_user_token": TENANT_USER_IDENTITY,
    "kb_tenant_admin_b_token": OTHER_TENANT_ADMIN_IDENTITY,
}


def get_db() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def authenticate_credentials(email: str, password: str) -> AuthIdentity | None:
    normalized_email = email.strip().lower()
    if normalized_email != ADMIN_IDENTITY.email:
        return None
    if not pwd_context.verify(password, ADMIN_PASSWORD_HASH):
        return None
    return ADMIN_IDENTITY


def get_current_identity(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> AuthIdentity:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    identity = _TOKEN_TO_IDENTITY.get(credentials.credentials)
    if identity is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")
    return identity


__all__ = [
    "AuthIdentity",
    "authenticate_credentials",
    "get_current_identity",
    "get_db",
    "require_tenant_admin",
    "require_platform_admin",
    "get_tenant_context",
]



def require_tenant_admin(identity: AuthIdentity = Depends(get_current_identity)) -> AuthIdentity:
    if "tenant-admin" not in identity.roles and "platform-admin" not in identity.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    return identity


def require_platform_admin(identity: AuthIdentity = Depends(get_current_identity)) -> AuthIdentity:
    if "platform-admin" not in identity.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    return identity


def get_tenant_context(identity: AuthIdentity = Depends(get_current_identity)) -> tuple[str, str]:
    if not identity.tenant_id or not identity.tenant_slug:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid tenant context")
    return identity.tenant_id, identity.tenant_slug
