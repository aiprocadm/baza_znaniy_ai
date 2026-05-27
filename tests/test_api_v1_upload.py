import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile
from types import ModuleType, SimpleNamespace
from typing import Mapping

import hashlib
import warnings

import pytest
import asyncio
import importlib.util
import pydantic
import sysconfig

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"builtin type swigvarlink has no __module__ attribute",
)

import app as _app_warning_filters  # noqa: F401 -- side effect: installs warning filters


if "fastapi" not in sys.modules:
    fastapi_root = Path(sysconfig.get_paths()["purelib"]) / "fastapi" / "__init__.py"
    if fastapi_root.exists():  # pragma: no cover - guard for optional dependency
        spec = importlib.util.spec_from_file_location(
            "fastapi",
            fastapi_root,
            submodule_search_locations=[str(fastapi_root.parent)],
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules["fastapi"] = module
        spec.loader.exec_module(module)

from fastapi import FastAPI


if "starlette.datastructures" not in sys.modules:
    starlette_module = ModuleType("starlette")
    datastructures_module = ModuleType("starlette.datastructures")

    class MutableHeaders(dict):
        def __init__(self, raw=None):
            super().__init__()
            self._raw: list[tuple[bytes, bytes]] = []
            if raw:
                for key, value in raw:
                    key_str = (
                        key.decode().lower()
                        if isinstance(key, (bytes, bytearray))
                        else str(key).lower()
                    )
                    value_str = (
                        value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
                    )
                    super().__setitem__(key_str, value_str)
                self._rebuild_raw()

        def _rebuild_raw(self) -> None:
            self._raw = [
                (key.encode("latin-1", "ignore"), value.encode("latin-1", "ignore"))
                for key, value in super().items()
            ]

        def __setitem__(self, key, value) -> None:  # type: ignore[override]
            key_str = str(key).lower()
            value_str = str(value)
            super().__setitem__(key_str, value_str)
            self._rebuild_raw()

        def pop(self, key, default=None):  # type: ignore[override]
            key_str = str(key).lower()
            result = super().pop(key_str, default)
            self._rebuild_raw()
            return result

        def items(self):  # type: ignore[override]
            return super().items()

        @property
        def raw(self) -> list[tuple[bytes, bytes]]:
            return list(self._raw)

    datastructures_module.MutableHeaders = MutableHeaders
    sys.modules["starlette"] = starlette_module
    sys.modules["starlette.datastructures"] = datastructures_module


def _coerce_form_entry(entry: object):
    from fastapi import UploadFile as FastAPIUploadFile  # type: ignore

    if isinstance(entry, FastAPIUploadFile):
        return entry
    if isinstance(entry, (list, tuple)):
        filename = entry[0] if entry else "uploaded"
        content = entry[1] if len(entry) > 1 else b""
        content_type = entry[2] if len(entry) > 2 else None
        return upload_utils.create_upload_file(filename, content, content_type)
    if isinstance(entry, dict):
        return upload_utils.create_upload_file(
            entry.get("filename"),
            entry.get("content", b""),
            entry.get("content_type"),
        )
    if isinstance(entry, str):
        return upload_utils.create_upload_file(entry, b"")
    return upload_utils.create_upload_file("uploaded", b"")


class _StubResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


def _invoke_post(app: FastAPI, path: str, *, files: list[tuple[str, object]]) -> _StubResponse:
    fastapi_module = sys.modules["fastapi"]
    route, params = app._find_route("POST", path)  # type: ignore[attr-defined]
    assert route is not None and params is not None, "route must exist"

    grouped: dict[str, list[object]] = {}
    for key, value in files:
        grouped.setdefault(key, []).append(value)

    body: dict[str, object] = {}
    for key, values in grouped.items():
        uploads = [_coerce_form_entry(item) for item in values]
        body[key] = uploads

    kwargs = fastapi_module._build_call_arguments(route.handler, body, params, app)  # type: ignore[attr-defined]
    limits_override = getattr(app, "dependency_overrides", {}).get(
        getattr(upload_module, "get_upload_limits", None)
    )
    if isinstance(kwargs.get("limits"), dict) and callable(limits_override):
        kwargs["limits"] = limits_override()
    result = route.handler(**kwargs)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    content = fastapi_module._serialise(result)  # type: ignore[attr-defined]
    if is_dataclass(content):
        content = asdict(content)
    return _StubResponse(route.status_code, content)


def _expected_storage_path(base: Path, tenant: str, payload: bytes, filename: str) -> Path:
    sha_value = hashlib.sha256(payload).hexdigest()
    suffix = Path(filename).suffix
    extension = suffix.lstrip(".")
    resolved_root = base.resolve()
    tenant_root = (resolved_root / tenant).resolve()
    bucket = tenant_root / sha_value[:2]
    suffix = f".{extension}" if extension else ""
    return bucket / f"{sha_value}{suffix}"


class _PrometheusMetric:
    def labels(self, **_: object) -> "_PrometheusMetric":
        return self

    def inc(self, *_: object, **__: object) -> None:
        return None

    def dec(self, *_: object, **__: object) -> None:
        return None

    def set(self, *_: object, **__: object) -> None:
        return None

    def observe(self, *_: object, **__: object) -> None:
        return None


if "prometheus_client" not in sys.modules:
    # Mirror the surface the production code consumes:
    # ``app/observability/metrics.py`` imports ``Counter``, ``Gauge``,
    # ``Histogram`` together, so an inline stub that omits ``Gauge`` (or
    # ``dec``/``set`` on the metric child) breaks the upload test's
    # transitive imports with the cryptic "cannot import name 'Gauge' from
    # '<unknown module name>'" — ``<unknown module name>`` because
    # ``SimpleNamespace`` has no ``__name__``.
    sys.modules["prometheus_client"] = SimpleNamespace(
        Counter=lambda *args, **kwargs: _PrometheusMetric(),
        Gauge=lambda *args, **kwargs: _PrometheusMetric(),
        Histogram=lambda *args, **kwargs: _PrometheusMetric(),
        CONTENT_TYPE_LATEST="text/plain",
        generate_latest=lambda: b"",
    )


class _Session:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        pass

    def __enter__(self):  # pragma: no cover - stub
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - stub
        return None


if "sqlmodel" not in sys.modules:
    sys.modules["sqlmodel"] = SimpleNamespace(
        Session=_Session,
        delete=lambda *args, **kwargs: None,
        select=lambda *args, **kwargs: None,
    )


_pydantic = sys.modules.get("pydantic", pydantic)
if _pydantic is not None and not hasattr(_pydantic, "field_validator"):

    def _field_validator(*args, **kwargs):  # pragma: no cover - stub
        def decorator(func):
            return func

        return decorator

    setattr(_pydantic, "field_validator", _field_validator)

if _pydantic is not None and not hasattr(_pydantic, "model_validator"):

    def _model_validator(*args, **kwargs):  # pragma: no cover - stub
        def decorator(func):
            return func

        return decorator

    setattr(_pydantic, "model_validator", _model_validator)


if "app.core.deps" not in sys.modules:
    deps_module = ModuleType("app.core.deps")

    @dataclass
    class UploadLimits:
        max_upload_mb: int = 1
        allowed_extensions: set[str] = None  # type: ignore[assignment]

        def __post_init__(self) -> None:
            exts = self.allowed_extensions or {"txt"}
            self.allowed_extensions = {ext.lower() for ext in exts}
            self.max_bytes = self.max_upload_mb * 1024 * 1024

        @property
        def max_size(self) -> int:
            return self.max_bytes

    deps_module.UploadLimits = UploadLimits
    deps_module.DEFAULT_ALLOWED_EXTENSIONS = frozenset({"txt", "md"})
    deps_module.get_data_dir = lambda: Path(".")
    deps_module.get_tenant = lambda: "default"
    deps_module.get_ingest_session = lambda: None
    deps_module.get_file_store = lambda: None
    deps_module.get_ingest_queue = lambda: None
    deps_module.get_ingest_service = lambda: None
    deps_module.get_upload_limits = UploadLimits
    sys.modules["app.core.deps"] = deps_module
else:
    from app.core.deps import UploadLimits  # type: ignore


upload_utils_path = Path(__file__).resolve().parents[1] / "app" / "api" / "upload_utils.py"
upload_utils_spec = importlib.util.spec_from_file_location(
    "app.api.upload_utils", upload_utils_path
)
assert upload_utils_spec and upload_utils_spec.loader
upload_utils = importlib.util.module_from_spec(upload_utils_spec)
sys.modules["app.api.upload_utils"] = upload_utils
upload_utils_spec.loader.exec_module(upload_utils)

models_module = sys.modules.setdefault("app.models", ModuleType("app.models"))
if not hasattr(models_module, "UploadResponse"):

    @dataclass
    class UploadResponse:
        file_id: str
        filename: str
        tenant: str
        status: str
        queued: bool = True

    models_module.UploadResponse = UploadResponse


user_module = sys.modules.setdefault("app.models.user", ModuleType("app.models.user"))
if not hasattr(user_module, "UserRecord"):

    @dataclass
    class UserRecord:
        id: int = 0

    user_module.UserRecord = UserRecord


ingest_module = sys.modules.setdefault("app.ingest.service", ModuleType("app.ingest.service"))
if not hasattr(ingest_module, "IngestService"):

    class IngestService:  # pragma: no cover - typing stub
        ...

    class IngestWorker:  # pragma: no cover - typing stub
        ...

    class IngestJob:  # pragma: no cover - typing stub
        ...

    class IngestQueueFullError(RuntimeError):  # pragma: no cover - typing stub
        ...

    ingest_module.IngestService = IngestService
    ingest_module.IngestWorker = IngestWorker
    ingest_module.IngestJob = IngestJob
    ingest_module.IngestQueueFullError = IngestQueueFullError


upload_path = Path(__file__).resolve().parents[1] / "app" / "api" / "v1" / "upload.py"
upload_spec = importlib.util.spec_from_file_location("app.api.v1.upload", upload_path)
assert upload_spec and upload_spec.loader
upload_module = importlib.util.module_from_spec(upload_spec)
sys.modules["app.api.v1.upload"] = upload_module
upload_spec.loader.exec_module(upload_module)

if "app.core.deps" in sys.modules:
    from app.core.deps import UploadLimits  # type: ignore
else:
    UploadLimits = sys.modules["app.core.deps"].UploadLimits  # type: ignore[attr-defined]


class _StubIngestService:
    def __init__(self) -> None:
        self.calls = []

    async def register_file(
        self,
        tenant: str,
        path: str,
        *,
        filename: str,
        size: int,
        mime_type: str,
    ):
        record = SimpleNamespace(
            id="file-1",
            filename=filename,
            status="queued",
            path=path,
        )
        self.calls.append(
            {
                "tenant": tenant,
                "path": path,
                "filename": filename,
                "size": size,
                "mime_type": mime_type,
            }
        )
        return record, True


_INVALID_CONTENT_TYPE_CANDIDATES = [
    "application/octet-stream",
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/html",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]


def _v1_invalid_mime_matrix() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    mapping = getattr(upload_module, "_ALLOWED_CONTENT_TYPES_BY_EXTENSION", {})
    for extension, allowed in sorted(mapping.items()):
        invalid_type = next(
            (
                candidate
                for candidate in _INVALID_CONTENT_TYPE_CANDIDATES
                if candidate not in allowed
            ),
            f"application/x-{extension}",
        )
        cases.append((extension, invalid_type))
    return cases


def _stub_request(headers: Mapping[str, str] | None = None) -> SimpleNamespace:
    normalized = {str(key).lower(): str(value) for key, value in (headers or {}).items()}
    return SimpleNamespace(headers=normalized)


def test_upload_file_accepts_uploadfile_instance(tmp_path: Path) -> None:
    limits = UploadLimits(max_upload_mb=1, allowed_extensions={"txt"})
    service = _StubIngestService()

    payload = b"hello world"
    stream = SpooledTemporaryFile(mode="w+b")
    stream.write(payload)
    stream.seek(0)
    upload = upload_module.UploadFile(
        filename="example.txt",
        file=stream,
        content_type="text/plain",
    )

    response = asyncio.run(
        upload_module.upload_file(
            file=[upload],
            files=None,
            limits=limits,
            data_dir=tmp_path,
            _=object(),
            tenant="default",
            ingest_service=service,  # type: ignore[arg-type]
            request=_stub_request(),
        )
    )

    assert response.file_id == "file-1"
    assert response.filename == "example.txt"
    assert response.tenant == "default"
    assert response.status == "queued"
    assert response.queued is True
    assert service.calls and service.calls[0]["size"] == len(payload)
    assert getattr(upload.file, "closed", False)


def test_upload_file_coerces_tuple_and_list_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    limits = UploadLimits(max_upload_mb=2, allowed_extensions={"txt", "md"})
    service = _StubIngestService()

    created_files = []
    closed_files = []

    class TrackingUploadFile:
        def __init__(self, *, filename=None, file=None, content_type=None, **kwargs):
            assert file is not None, "file argument is required"
            assert "content" not in kwargs or kwargs["content"] in {None, b""}
            self.filename = filename
            self.file = file
            self.content_type = content_type
            created_files.append(file)

        async def read(self):
            return self.file.read()

        async def close(self):
            closed_files.append(self.file)
            if hasattr(self.file, "close") and not getattr(self.file, "closed", False):
                self.file.close()

    monkeypatch.setattr(upload_utils, "UploadFile", TrackingUploadFile)

    response = asyncio.run(
        upload_module.upload_file(
            file=[("ignored.bin", b"bin")],
            files=[("final.txt", b"tuple-bytes", "text/plain")],
            limits=limits,
            data_dir=tmp_path,
            _=object(),
            tenant="acme",
            ingest_service=service,  # type: ignore[arg-type]
            request=_stub_request(),
        )
    )

    assert response.filename == "final.txt"
    assert service.calls[0]["mime_type"] == "text/plain"
    assert all(getattr(file, "closed", False) for file in created_files)
    assert closed_files and all(file in closed_files for file in created_files)


def test_upload_file_tuple_input_preserves_payload(tmp_path: Path) -> None:
    limits = UploadLimits(max_upload_mb=1, allowed_extensions={"txt"})
    service = _StubIngestService()

    payload = b"tuple-content"
    stream = SpooledTemporaryFile(mode="w+b")
    stream.write(payload)
    stream.seek(len(payload))

    response = asyncio.run(
        upload_module.upload_file(
            file=[("tuple.txt", stream, "text/plain")],
            files=None,
            limits=limits,
            data_dir=tmp_path,
            _=object(),
            tenant="tuple-tenant",
            ingest_service=service,  # type: ignore[arg-type]
            request=_stub_request(),
        )
    )

    assert response.filename == "tuple.txt"
    recorded = service.calls[0]
    expected_path = _expected_storage_path(tmp_path, "tuple-tenant", payload, recorded["filename"])
    stored_path = Path(recorded["path"])
    assert stored_path == expected_path
    assert stored_path.exists()
    assert stored_path.read_bytes() == payload
    assert recorded["size"] == len(payload)


def test_upload_file_list_input_preserves_payload(tmp_path: Path) -> None:
    limits = UploadLimits(max_upload_mb=1, allowed_extensions={"txt"})
    service = _StubIngestService()

    payload = b"list-content"
    stream = SpooledTemporaryFile(mode="w+b")
    stream.write(payload)
    stream.seek(len(payload))

    response = asyncio.run(
        upload_module.upload_file(
            file=[["list.txt", stream, "text/plain"]],
            files=None,
            limits=limits,
            data_dir=tmp_path,
            _=object(),
            tenant="list-tenant",
            ingest_service=service,  # type: ignore[arg-type]
            request=_stub_request(),
        )
    )

    assert response.filename == "list.txt"
    recorded = service.calls[0]
    expected_path = _expected_storage_path(tmp_path, "list-tenant", payload, recorded["filename"])
    stored_path = Path(recorded["path"])
    assert stored_path == expected_path
    assert stored_path.exists()
    assert stored_path.read_bytes() == payload
    assert recorded["size"] == len(payload)


def test_upload_file_dict_input_preserves_payload(tmp_path: Path) -> None:
    limits = UploadLimits(max_upload_mb=1, allowed_extensions={"txt"})
    service = _StubIngestService()

    payload = b"dict-content"
    entry = {
        "filename": "dict.txt",
        "content": payload,
        "content_type": "text/plain",
    }

    response = asyncio.run(
        upload_module.upload_file(
            file=[entry],
            files=None,
            limits=limits,
            data_dir=tmp_path,
            _=object(),
            tenant="dict-tenant",
            ingest_service=service,  # type: ignore[arg-type]
            request=_stub_request(),
        )
    )

    assert response.filename == "dict.txt"
    recorded = service.calls[0]
    expected_path = _expected_storage_path(tmp_path, "dict-tenant", payload, recorded["filename"])
    stored_path = Path(recorded["path"])
    assert stored_path == expected_path
    assert stored_path.exists()
    assert stored_path.read_bytes() == payload
    assert recorded["size"] == len(payload)


def test_upload_file_rejects_payload_exceeding_limits(tmp_path: Path) -> None:
    limits = UploadLimits(max_upload_mb=1, allowed_extensions={"txt"})
    service = _StubIngestService()

    payload = b"x" * (limits.max_size + 1)
    stream = SpooledTemporaryFile(mode="w+b")
    stream.write(payload)
    stream.seek(0)

    upload = upload_module.UploadFile(
        filename="too-big.txt",
        file=stream,
        content_type="text/plain",
    )

    with pytest.raises(upload_module.HTTPException) as excinfo:
        asyncio.run(
            upload_module.upload_file(
                file=[upload],
                files=None,
                limits=limits,
                data_dir=tmp_path,
                _=object(),
                tenant="default",
                ingest_service=service,  # type: ignore[arg-type]
                request=_stub_request(),
            )
        )

    assert excinfo.value.status_code == upload_module.HTTP_CONTENT_TOO_LARGE
    assert excinfo.value.detail == "UPLOAD_TOO_LARGE"
    assert service.calls == []


@pytest.mark.parametrize("extension, invalid_type", _v1_invalid_mime_matrix())
def test_upload_file_rejects_invalid_content_type(
    tmp_path: Path, extension: str, invalid_type: str
) -> None:
    limits = UploadLimits(max_upload_mb=1, allowed_extensions={extension})
    service = _StubIngestService()

    payload = b"valid"
    stream = SpooledTemporaryFile(mode="w+b")
    stream.write(payload)
    stream.seek(0)

    upload = upload_module.UploadFile(
        filename=f"document.{extension}",
        file=stream,
        content_type=invalid_type,
    )

    with pytest.raises(upload_module.HTTPException) as excinfo:
        asyncio.run(
            upload_module.upload_file(
                file=[upload],
                files=None,
                limits=limits,
                data_dir=tmp_path,
                _=object(),
                tenant="default",
                ingest_service=service,  # type: ignore[arg-type]
                request=_stub_request(),
            )
        )

    assert excinfo.value.status_code == getattr(upload_module, "_UNSUPPORTED_MEDIA_TYPE", 415)
    assert excinfo.value.detail == "UPLOAD_INVALID_TYPE"
    assert service.calls == []


@pytest.mark.skip(
    reason=(
        "_invoke_post helper relies on internals of the in-tree fastapi "
        "test stub (`_build_call_arguments`, `_find_route`). On CI (real "
        "fastapi from PyPI) it fails with `fastapi_inner_astack not found "
        "in request scope`; locally (real fastapi too on py3.13) the "
        "helper doesn't apply the auth dependency_overrides correctly and "
        "the call surfaces a 401. The behaviour being tested (router "
        "accepts multiple file inputs) is already covered by the v1 "
        "TestClient tests; re-enable once the helper is rewritten on top "
        "of fastapi.testclient.TestClient."
    )
)
def test_upload_endpoint_handles_multiple_formats(tmp_path: Path) -> None:
    limits = UploadLimits(max_upload_mb=2, allowed_extensions={"pdf", "md", "txt"})
    service = _StubIngestService()

    app = FastAPI()
    app.include_router(upload_module.router)

    app.dependency_overrides[upload_module.get_upload_limits] = lambda: limits
    app.dependency_overrides[upload_module.get_data_dir] = lambda: tmp_path
    app.dependency_overrides[upload_module.get_current_active_user] = lambda: object()
    app.dependency_overrides[upload_module.ensure_tenant_access] = lambda: "tenant-x"
    app.dependency_overrides[upload_module.get_ingest_service] = lambda: service

    pdf_payload = b"%PDF-1.7\n"
    markdown_payload = b"# Heading\n"

    files = [
        ("file", ("ignored.exe", b"MZ" * 10, "application/octet-stream")),
        ("files", ("manual.pdf", pdf_payload, "application/pdf")),
        ("files", ("notes.md", markdown_payload, "text/markdown")),
    ]

    response = _invoke_post(app, "/upload", files=files)

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "manual.pdf"
    assert body["tenant"] == "tenant-x"

    assert service.calls, "register_file should be invoked"
    recorded = service.calls[0]
    assert recorded["mime_type"] == "application/pdf"
    expected_path = _expected_storage_path(tmp_path, "tenant-x", pdf_payload, recorded["filename"])
    stored_path = Path(recorded["path"])
    assert stored_path == expected_path
    assert stored_path.exists()
    assert stored_path.read_bytes() == pdf_payload


def test_upload_does_not_mutate_original_uploadfile(tmp_path: Path) -> None:
    limits = UploadLimits(max_upload_mb=1, allowed_extensions={"txt"})
    service = _StubIngestService()

    payload = b"immutable"
    stream = SpooledTemporaryFile(mode="w+b")
    stream.write(payload)
    stream.seek(0)

    headers = upload_module.MutableHeaders()
    headers["content-disposition"] = 'form-data; name="file"; filename="immutable.txt"'

    original = upload_module.FastAPIUploadFile(
        file=stream,
        filename="",
        headers=headers,
    )
    headers["content-type"] = "text/plain"

    response = asyncio.run(
        upload_module.upload_file(
            file=[original],
            files=None,
            limits=limits,
            data_dir=tmp_path,
            _=object(),
            tenant="immutable",
            ingest_service=service,  # type: ignore[arg-type]
            request=_stub_request(),
        )
    )

    assert response.filename == "immutable.txt"
    assert original.filename == ""
    recorded = service.calls[0]
    expected_path = _expected_storage_path(tmp_path, "immutable", payload, recorded["filename"])
    stored_path = Path(recorded["path"])
    assert stored_path == expected_path
    assert stored_path.read_bytes() == payload
