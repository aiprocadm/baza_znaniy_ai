from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.auth import _hash_api_key, get_subject_attribution


class _Headers(dict):
    def get(self, key, default=None):
        return super().get(key, super().get(key.lower(), default))


class _Request:
    def __init__(self, headers: dict[str, str]):
        self.headers = _Headers(headers)


class _Result:
    def __init__(self, value):
        self._value = value

    def first(self):
        return self._value


class _Session:
    def __init__(self, value):
        self._value = value

    def exec(self, _query):
        return _Result(self._value)


class _ApiKey:
    id = 7


def test_hash_api_key_requires_salt(monkeypatch):
    monkeypatch.delenv("API_KEY_HASH_SALT", raising=False)
    with pytest.raises(HTTPException):
        _hash_api_key("abc")


def test_subject_attribution_resolves_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY_HASH_SALT", "salt")
    subject = get_subject_attribution(
        request=_Request({"x-api-key": "k1", "x-tenant": "tenant-a"}),
        session=_Session(_ApiKey()),
    )
    assert subject.subject_type == "api_key"
    assert subject.tenant == "tenant-a"
    assert subject.subject_id == "7"


def test_subject_attribution_rejects_invalid_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY_HASH_SALT", "salt")
    with pytest.raises(HTTPException):
        get_subject_attribution(
            request=_Request({"x-api-key": "k2", "x-tenant": "tenant-a"}),
            session=_Session(None),
        )
