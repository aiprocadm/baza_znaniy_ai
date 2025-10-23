from __future__ import annotations

import importlib
import io
from http import HTTPStatus
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, assume, settings
from schemathesis import openapi
from sqlmodel import Session, select

from backend.app.main import create_app as create_backend_app
from tests.service_stubs import install_service_stubs


def _flatten_aliases_compat(source: Any) -> list[str]:
    """Return aliases from ``pydantic.AliasChoices`` across versions."""

    if source is None or source is Ellipsis:
        return []

    choices: list[str] = []

    if isinstance(source, (str, bytes)):
        return [source.decode("utf-8") if isinstance(source, bytes) else source]

    if hasattr(source, "choices"):
        raw = getattr(source, "choices")
        if isinstance(raw, (list, tuple, set)):
            for item in raw:
                choices.extend(_flatten_aliases_compat(item))
        else:
            try:
                iterator = iter(raw)
            except TypeError:
                return [str(raw)]
            else:
                for item in iterator:
                    choices.extend(_flatten_aliases_compat(item))
        return choices

    try:
        iterator = iter(source)
    except TypeError:
        return [str(source)]
    return [
        alias
        for item in iterator
        for alias in _flatten_aliases_compat(item)
    ] or [str(source)]

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "openapi.yaml"
SCHEMA = openapi.from_path(str(SCHEMA_PATH))


class SchemathesisTestClient(TestClient):
    """FastAPI TestClient variant compatible with Schemathesis transports."""

    def request(self, method: str, url: str, **kwargs: Any):  # type: ignore[override]
        kwargs.pop("verify", None)
        response = super().request(method, url, **kwargs)
        if not hasattr(response, "raw"):
            response.raw = self._RawBuffer(response.content, response.headers)  # type: ignore[attr-defined]
        if not hasattr(response, "reason"):
            try:
                response.reason = HTTPStatus(response.status_code).phrase  # type: ignore[attr-defined]
            except ValueError:
                response.reason = ""
        return response

    class _RawBuffer(io.BytesIO):
        def __init__(self, content: bytes, headers: Mapping[str, Any]) -> None:
            super().__init__(content)
            self.headers = SchemathesisTestClient._RawHeaders(headers)
            self.version = 11

    class _RawHeaders(dict[str, list[Any]]):
        def __init__(self, source: Mapping[str, Any]) -> None:
            super().__init__()
            for key, value in source.items():
                values: list[Any]
                if isinstance(value, (list, tuple)):
                    values = list(value)
                else:
                    values = [value]
                self[key.lower()] = values

        def getlist(self, name: str) -> list[Any]:
            return self.get(name.lower(), [])

        def get_list(self, name: str) -> list[Any]:
            return self.getlist(name)

_ENDPOINT_TARGETS: dict[str, dict[str, str]] = {
    "/auth/login": {"app": "core", "path": "/api/v1/auth/login"},
    "/files/upload": {"app": "core", "path": "/api/v1/upload"},
    "/documents/generate": {"app": "backend", "path": "/api/v1/documents/generate"},
    "/packs/run": {"app": "backend", "path": "/api/v1/packs/run"},
}


@pytest.fixture()
def core_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Create a FastAPI test client for the core service with lightweight stubs."""

    install_service_stubs()

    env_overrides = {
        "DATA_DIR": str(tmp_path / "data"),
        "DB_URL": f"sqlite:///{tmp_path / 'core.db'}",
        "SECRET_KEY": "test-secret",
        "JWT_ALGORITHM": "HS256",
        "ACCESS_TOKEN_EXPIRE_MINUTES": "5",
        "RERANK_ENABLED": "0",
        "VECTOR_BACKEND": "qdrant",
        "INGEST_AUTOSTART_WORKER": "false",
        "INGEST_USE_LOCAL_QUEUE": "true",
        "CELERY_TASK_ALWAYS_EAGER": "1",
        "QDRANT_URL": "http://localhost:6333",
    }
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)

    from pydantic import AliasChoices as PydanticAliasChoices

    if not callable(getattr(PydanticAliasChoices, "__iter__", None)):
        def _aliaschoices_iter(self: Any) -> Iterator[Any]:
            raw = getattr(self, "choices", ())
            if isinstance(raw, (list, tuple, set)):
                return iter(raw)
            if isinstance(raw, (str, bytes)):
                return iter([raw.decode("utf-8") if isinstance(raw, bytes) else raw])
            try:
                return iter(raw)
            except TypeError:
                return iter([raw])

        monkeypatch.setattr(PydanticAliasChoices, "__iter__", _aliaschoices_iter, raising=False)

    from app.core import config as config_module
    from app.models import file as file_models
    from app.retriever import get_vector_store

    config_module.get_settings.cache_clear()
    file_models.get_engine.cache_clear()
    get_vector_store.cache_clear()

    import app.main as app_main

    app_main = importlib.reload(app_main)
    config_module = importlib.reload(config_module)
    monkeypatch.setattr(config_module, "_flatten_aliases", _flatten_aliases_compat)
    config_module.get_settings.cache_clear()

    app = app_main.app

    client = SchemathesisTestClient(app, raise_server_exceptions=False)

    from app.models.tenant import TenantRecord
    from app.models.user import UserRecord, UserRole
    from app.security import hash_password

    service = app.state.ingest_service
    with Session(service.engine) as session:
        tenant = session.exec(select(TenantRecord).where(TenantRecord.slug == "default")).first()
        if tenant is None:
            tenant = TenantRecord(slug="default", name="Default", is_active=True)
            session.add(tenant)
            session.commit()
        user = session.exec(select(UserRecord).where(UserRecord.email == "admin@example.com")).first()
        if user is None:
            user = UserRecord(
                email="admin@example.com",
                full_name="Admin",
                tenant_slug=tenant.slug,
                role=UserRole.ADMIN,
                is_active=True,
                hashed_password=hash_password("secret"),
            )
            session.add(user)
            session.commit()

    try:
        yield client
    finally:
        client.close()
        app.dependency_overrides.clear()
        file_models.get_engine.cache_clear()
        config_module.get_settings.cache_clear()
        get_vector_store.cache_clear()


@pytest.fixture()
def backend_client() -> Iterator[TestClient]:
    """Create a FastAPI test client for the document backend service."""

    app = create_backend_app()
    client = SchemathesisTestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        client.close()


@SCHEMA.parametrize()
@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_openapi_contract(case, core_client: TestClient, backend_client: TestClient) -> None:
    """Validate service responses against the OpenAPI contract using Schemathesis."""

    mapping = _ENDPOINT_TARGETS.get(case.path)
    assert mapping is not None, f"Unexpected path {case.path!r} in contract test"

    target_client = core_client if mapping["app"] == "core" else backend_client
    case.path = mapping["path"]
    assume(case.method is None or case.method.lower() == "post")

    if mapping["app"] == "core" and case.path.endswith("/auth/login"):
        case.media_type = "application/json"
        case.headers = {"Content-Type": "application/json", **(case.headers or {})}
        case.body = {"email": "admin@example.com", "password": "invalid"}
    elif mapping["app"] == "backend" and case.path.endswith("/documents/generate"):
        case.media_type = "application/json"
        case.headers = {"Content-Type": "application/json", **(case.headers or {})}
        case.body = {"template_id": "missing", "document_name": "Test", "context": {}}
    elif mapping["app"] == "backend" and case.path.endswith("/packs/run"):
        case.media_type = "application/json"
        case.headers = {"Content-Type": "application/json", **(case.headers or {})}
        case.body = {"pack_id": 999999}

    response = case.call_and_validate(session=target_client, base_url=str(target_client.base_url))
    assert response is not None
