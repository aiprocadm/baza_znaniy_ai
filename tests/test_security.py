from __future__ import annotations

import importlib
import sys
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture()
def security(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "unit-test-secret")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "5")

    if "app.security" in sys.modules:
        module = importlib.reload(sys.modules["app.security"])
    else:
        module = importlib.import_module("app.security")
    return module


def test_hash_and_verify_password_success(security):
    hashed = security.hash_password("password123")

    assert hashed != "password123"
    assert security.verify_password("password123", hashed)


def test_verify_password_rejects_invalid_input(security):
    hashed = security.hash_password("correct horse battery staple")

    assert not security.verify_password("wrong password", hashed)
    assert not security.verify_password("irrelevant", "not-a-valid-hash")


def test_create_access_token_includes_expiry(security):
    now = datetime.now(timezone.utc)
    token = security.create_access_token(
        {"sub": "user-123"}, expires_delta=timedelta(minutes=1)
    )

    payload = security.decode_token(token)
    assert payload["sub"] == "user-123"

    expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    assert timedelta(seconds=0) <= expires_at - now <= timedelta(minutes=1, seconds=5)


def test_decode_token_rejects_invalid_tokens(security):
    with pytest.raises(security.InvalidTokenError):
        security.decode_token("this-is-not-a-token")

    expired_token = security.create_access_token(
        {"sub": "user-123"}, expires_delta=timedelta(seconds=-1)
    )
    with pytest.raises(security.InvalidTokenError):
        security.decode_token(expired_token)

