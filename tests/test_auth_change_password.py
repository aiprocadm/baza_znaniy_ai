"""Tests for the password change flow using the shared auth router."""

from __future__ import annotations

import sys
from pathlib import Path


def test_user_with_mandatory_password_change_can_update(monkeypatch, tmp_path):
    db_path = tmp_path / "data" / "app.sqlite"
    files_root = tmp_path / "files"

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("FILES_ROOT", str(files_root))
    monkeypatch.setenv("CHAT_MEMORY_ENABLED", "false")
    monkeypatch.setenv("APP_SECRET", "test-secret")

    for module in [
        "app.db.session",
        "app.db.models",
        "app.db.seed",
        "app.auth",
    ]:
        sys.modules.pop(module, None)

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from importlib import import_module
    import pytest
    from fastapi.security import HTTPAuthorizationCredentials

    session_module = import_module("app.db.session")
    models = import_module("app.db.models")
    seed = import_module("app.db.seed")
    auth = import_module("app.auth")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    files_root.mkdir(parents=True, exist_ok=True)

    session_module.init_db()

    with session_module.SessionLocal() as session:
        user = models.User(
            login="user@example.com",
            password_hash=seed.hash_password("initialPass1"),
            role=models.UserRole.STAFF,
            must_change_password=True,
        )
        session.add(user)
        session.commit()

    with session_module.SessionLocal() as session:
        login_result = auth.login(
            auth.LoginRequest(login="user@example.com", password="initialPass1"),
            session=session,
        )

    assert login_result.must_change_password is True
    token = login_result.token

    with session_module.SessionLocal() as session:
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        current_user = auth.get_current_user(session=session, credentials=credentials)
        change_result = auth.change_password(
            auth.ChangeCredentialsRequest(new_password="ChangedPass2"),
            session=session,
            user=current_user,
        )

    assert change_result.must_change_password is False
    assert change_result.login == "user@example.com"
    assert change_result.token

    with session_module.SessionLocal() as session:
        with pytest.raises(auth.HTTPException) as excinfo:
            auth.login(
                auth.LoginRequest(login="user@example.com", password="initialPass1"),
                session=session,
            )

    assert excinfo.value.status_code == 401

    with session_module.SessionLocal() as session:
        login_after_change = auth.login(
            auth.LoginRequest(login="user@example.com", password="ChangedPass2"),
            session=session,
        )

    assert login_after_change.must_change_password is False
