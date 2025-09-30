"""Security helpers for password hashing and token management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.exc import UnknownHashError

from app.core.config import get_settings

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

_initial_settings = get_settings()
SECRET_KEY = _initial_settings.secret_key
ALGORITHM = _initial_settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = _initial_settings.access_token_expire_minutes


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
    settings = get_settings()
    expire_minutes = globals().get("ACCESS_TOKEN_EXPIRE_MINUTES", settings.access_token_expire_minutes)
    expire_delta = expires_delta or timedelta(minutes=expire_minutes)
    expire = datetime.now(timezone.utc) + expire_delta
    to_encode.update({"exp": expire})
    secret = globals().get("SECRET_KEY", settings.secret_key)
    algorithm = globals().get("ALGORITHM", settings.jwt_algorithm)
    return jwt.encode(to_encode, secret, algorithm=algorithm)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode a JWT and return its payload, raising on errors."""

    try:
        settings = get_settings()
        secret = globals().get("SECRET_KEY", settings.secret_key)
        algorithm = globals().get("ALGORITHM", settings.jwt_algorithm)
        return jwt.decode(token, secret, algorithms=[algorithm])
    except JWTError as exc:
        raise InvalidTokenError("Could not validate credentials") from exc

