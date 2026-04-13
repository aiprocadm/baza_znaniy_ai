from __future__ import annotations

from dataclasses import dataclass
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
    roles: tuple[str, ...]


ADMIN_IDENTITY = AuthIdentity(
    user_id="u_admin",
    email="admin@kb.ai",
    roles=("admin",),
)
ADMIN_PASSWORD_HASH = pwd_context.hash("secret")

_TOKEN_TO_IDENTITY: dict[str, AuthIdentity] = {
    "kb_admin_token": ADMIN_IDENTITY,
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


def require_admin(identity: AuthIdentity = Depends(get_current_identity)) -> AuthIdentity:
    if "admin" not in identity.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
    return identity


__all__ = [
    "AuthIdentity",
    "authenticate_credentials",
    "get_current_identity",
    "get_db",
    "require_admin",
]
