import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import SpooledTemporaryFile
from types import ModuleType, SimpleNamespace

import pytest
import asyncio
import importlib.util
import pydantic


class _PrometheusMetric:
    def labels(self, **_: object) -> "_PrometheusMetric":
        return self

    def inc(self, *_: object, **__: object) -> None:
        return None

    def observe(self, *_: object, **__: object) -> None:
        return None


if "prometheus_client" not in sys.modules:
    sys.modules["prometheus_client"] = SimpleNamespace(
        Counter=lambda *args, **kwargs: _PrometheusMetric(),
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


if "app.core.auth" not in sys.modules:
    auth_module = ModuleType("app.core.auth")
    auth_module.ensure_tenant_access = lambda: "default"
    auth_module.get_current_active_user = lambda: object()
    sys.modules["app.core.auth"] = auth_module


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
    deps_module.get_ingest_service = lambda: None
    deps_module.get_upload_limits = UploadLimits
    sys.modules["app.core.deps"] = deps_module
else:
    from app.core.deps import UploadLimits  # type: ignore


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

    ingest_module.IngestService = IngestService
    ingest_module.IngestWorker = IngestWorker
    ingest_module.IngestJob = IngestJob


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
        )
    )

    assert response.file_id == "file-1"
    assert response.filename == "example.txt"
    assert response.tenant == "default"
    assert response.status == "queued"
    assert response.queued is True
    assert service.calls and service.calls[0]["size"] == len(payload)
    assert getattr(upload.file, "closed", False)


def test_upload_file_coerces_tuple_and_list_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    monkeypatch.setattr(upload_module, "UploadFile", TrackingUploadFile)

    response = asyncio.run(
        upload_module.upload_file(
            file=[("ignored.bin", b"bin")],
            files=[("final.txt", b"tuple-bytes", "text/plain")],
            limits=limits,
            data_dir=tmp_path,
            _=object(),
            tenant="acme",
            ingest_service=service,  # type: ignore[arg-type]
        )
    )

    assert response.filename == "final.txt"
    assert service.calls[0]["mime_type"] == "text/plain"
    assert all(getattr(file, "closed", False) for file in created_files)
    assert closed_files and all(file in closed_files for file in created_files)
