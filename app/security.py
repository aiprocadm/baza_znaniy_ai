"""Security helpers for password hashing and token management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from typing import Any, Dict, Tuple

from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.exc import UnknownHashError

from app.core.config import get_settings

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class InvalidTokenError(Exception):
    """Raised when a JWT cannot be decoded or validated."""


def _coerce_exp_to_numeric(value: Any) -> int | float:
    """Convert an ``exp`` claim into a numeric timestamp."""

    if isinstance(value, datetime):
        timestamp = value.timestamp()
    elif isinstance(value, (int, float)):
        timestamp = float(value)
    elif isinstance(value, str):
        try:
            timestamp = float(value)
        except ValueError:
            try:
                timestamp = datetime.fromisoformat(value).timestamp()
            except ValueError as exc:
                raise InvalidTokenError("Could not validate credentials") from exc
    else:
        raise InvalidTokenError("Could not validate credentials")

    numeric = float(timestamp)
    return int(numeric) if numeric.is_integer() else numeric


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


def _jwt_parameters() -> Tuple[str, str, int]:
    """Return the secret, algorithm and default expiry from settings."""

    settings = get_settings()
    return (
        settings.secret_key,
        settings.jwt_algorithm,
        int(settings.access_token_expire_minutes),
    )


def create_access_token(
    data: Dict[str, Any], *, expires_delta: timedelta | None = None
) -> str:
    """Create a signed JWT carrying the provided payload."""

    to_encode = dict(data) if data is not None else {}
    secret, algorithm, default_expiry_minutes = _jwt_parameters()
    expire_delta = expires_delta or timedelta(minutes=default_expiry_minutes)
    expire = datetime.now(timezone.utc) + expire_delta
    to_encode["exp"] = _coerce_exp_to_numeric(expire)
    issuer = os.getenv("JWT_ISSUER", "baza-znaniy-ai")
    audience = os.getenv("JWT_AUDIENCE", "baza-znaniy-clients")
    to_encode.setdefault("iss", issuer)
    to_encode.setdefault("aud", audience)
    return jwt.encode(to_encode, secret, algorithm=algorithm)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode a JWT and return its payload, raising on errors."""

    try:
        secret, algorithm, _ = _jwt_parameters()
        payload = jwt.decode(token, secret, algorithms=[algorithm])
        exp_value = payload.get("exp")
        if exp_value is not None:
            payload["exp"] = _coerce_exp_to_numeric(exp_value)
        return payload
    except JWTError as exc:
        raise InvalidTokenError("Could not validate credentials") from exc
