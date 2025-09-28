from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import models
from app.db.seed import ensure_default_admin, hash_password, verify_password
from app.db.session import SessionLocal, get_session

ALGORITHM = "HS256"
SECRET_KEY = os.getenv("APP_SECRET", "dev")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)


class TokenResponse(BaseModel):
    token: str
    role: models.UserRole
    must_change_password: bool
    login: str


class LoginRequest(BaseModel):
    login: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class ChangeCredentialsRequest(BaseModel):
    new_login: str | None = Field(default=None, min_length=1)
    new_password: str = Field(..., min_length=8)


UserSession = Annotated[Session, Depends(get_session)]


def create_access_token(*, user: models.User) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user.id), "role": user.role.value, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict[str, str]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:  # pragma: no cover - defensive branch
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="INVALID_TOKEN") from exc


def get_current_user(
    session: UserSession,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> models.User:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="AUTH_REQUIRED")

    payload = _decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="INVALID_TOKEN")

    user = session.get(models.User, int(user_id))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="USER_NOT_FOUND")

    return user


def require_active_user(user: models.User = Depends(get_current_user)) -> models.User:
    if user.must_change_password:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="PASSWORD_CHANGE_REQUIRED")
    return user


def require_admin(user: models.User = Depends(require_active_user)) -> models.User:
    if user.role != models.UserRole.ADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="ADMIN_REQUIRED")
    return user


def require_staff(user: models.User = Depends(require_active_user)) -> models.User:
    if user.role not in {models.UserRole.STAFF, models.UserRole.ADMIN}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="STAFF_REQUIRED")
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, session: UserSession) -> TokenResponse:
    stmt = select(models.User).where(models.User.username == payload.login)
    user = session.scalars(stmt).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="INVALID_CREDENTIALS")

    token = create_access_token(user=user)
    return TokenResponse(
        token=token,
        role=user.role,
        must_change_password=user.must_change_password,
        login=user.login,
    )


@router.post("/change-password", response_model=TokenResponse)
def change_password(
    payload: ChangeCredentialsRequest,
    session: UserSession,
    user: models.User = Depends(get_current_user),
) -> TokenResponse:
    if payload.new_login and payload.new_login != user.login:
        stmt = select(models.User).where(models.User.username == payload.new_login)
        existing = session.scalars(stmt).first()
        if existing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="LOGIN_TAKEN")
        user.login = payload.new_login

    user.password_hash = hash_password(payload.new_password)
    user.must_change_password = False
    try:
        session.add(user)
        session.commit()
    except IntegrityError as exc:  # pragma: no cover
        session.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="UPDATE_FAILED") from exc

    session.refresh(user)
    token = create_access_token(user=user)
    return TokenResponse(
        token=token,
        role=user.role,
        must_change_password=user.must_change_password,
        login=user.login,
    )


def setup_defaults() -> None:
    """Ensure the database schema and default admin user exist."""
    from app.db.session import init_db

    init_db()
    with SessionLocal() as session:
        ensure_default_admin(session)
