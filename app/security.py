"""Security helpers for password hashing and token management."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.exc import UnknownHashError

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


class InvalidTokenError(Exception):
    """Raised when a JWT cannot be decoded or validated."""


def hash_password(password: str) -> str:
    """Hash a plaintext password using argon2."""

    if not isinstance(password, str):
        raise TypeError("password must be a string")
    if password == "":
        raise ValueError("password must not be empty")
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Validate a plaintext password against a stored hash."""

    try:
        return _pwd_context.verify(plain_password, hashed_password)
    except (ValueError, UnknownHashError):
        return False


def create_access_token(
    data: Dict[str, Any], *, expires_delta: timedelta | None = None
) -> str:
    """Create a signed JWT carrying the provided payload."""

    to_encode = dict(data) if data is not None else {}
    expire_delta = expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire = datetime.now(timezone.utc) + expire_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode a JWT and return its payload, raising on errors."""

    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise InvalidTokenError("Could not validate credentials") from exc

