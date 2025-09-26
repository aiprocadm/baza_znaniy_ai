"""Security helpers for password hashing and JWT tokens."""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from argon2 import PasswordHasher, Type
from argon2.exceptions import VerificationError
from jose import JWTError, jwt

SECRET_KEY = os.getenv("APP_SECRET", "dev")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

_password_hasher = PasswordHasher(type=Type.ID)


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _password_hasher.verify(password_hash, password)
    except VerificationError:
        return False


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:  # pragma: no cover - defensive
        raise ValueError("INVALID_TOKEN") from exc


__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_token",
]
