from __future__ import annotations

import os
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import models

DEFAULT_ADMIN_LOGIN = os.getenv("DEFAULT_ADMIN_LOGIN", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin")

_password_hasher: Optional[PasswordHasher] = None


def get_password_hasher() -> PasswordHasher:
    global _password_hasher
    if _password_hasher is None:
        _password_hasher = PasswordHasher(time_cost=2, memory_cost=102400, parallelism=8)
    return _password_hasher


def hash_password(password: str) -> str:
    return get_password_hasher().hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return get_password_hasher().verify(password_hash, password)
    except (VerifyMismatchError, VerificationError):
        return False


def ensure_default_admin(session: Session) -> None:
    stmt = select(models.User).where(models.User.username == DEFAULT_ADMIN_LOGIN)
    user = session.scalars(stmt).first()
    if user:
        return

    admin = models.User(
        login=DEFAULT_ADMIN_LOGIN,
        password_hash=hash_password(DEFAULT_ADMIN_PASSWORD),
        role=models.UserRole.ADMIN,
        must_change_password=True,
    )
    session.add(admin)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
