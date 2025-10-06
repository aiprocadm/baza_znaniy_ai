import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile, status
from starlette.datastructures import MutableHeaders
from sqlmodel import Session, select

from app.api.v1.ingest import ingest_file
from app.api.v1.upload import upload_file
from app.core.config import get_settings
from app.core.deps import UploadLimits
from app.ingest.service import IngestQueueFullError, IngestService
from app.models import IngestRequest
from app.models import file as file_models
from app.models.file import FileRecord


@pytest.fixture
def configured_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> IngestService:
    db_path = tmp_path / "ingest.db"
    monkeypatch.setenv("DB_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("INGEST_QUEUE_SIZE", "1")
    get_settings.cache_clear()
    file_models.get_engine.cache_clear()
    service = IngestService(max_retries=0, backoff_seconds=0)
    yield service
    file_models.get_engine.cache_clear()
    get_settings.cache_clear()


def _write_sample(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_register_file_rolls_back_when_queue_full(
    configured_service: IngestService, tmp_path: Path
) -> None:
    async def scenario() -> None:
        first = _write_sample(tmp_path / "first.txt", "alpha")
        second = _write_sample(tmp_path / "second.txt", "bravo")

        record, queued = await configured_service.register_file(
            "tenant",
            str(first),
            filename="first.txt",
            size=first.stat().st_size,
            mime_type="text/plain",
        )
        assert queued is True
        assert configured_service.queue.full() is True

        with pytest.raises(IngestQueueFullError):
            await configured_service.register_file(
                "tenant",
                str(second),
                filename="second.txt",
                size=second.stat().st_size,
                mime_type="text/plain",
            )

        with Session(configured_service.engine) as session:
            files = session.exec(select(FileRecord)).all()
            assert len(files) == 1
            assert files[0].filename == record.filename

    asyncio.run(scenario())


def test_upload_endpoint_returns_429_on_queue_overflow(
    configured_service: IngestService, tmp_path: Path
) -> None:
    async def scenario() -> None:
        first = _write_sample(tmp_path / "first.txt", "alpha")
        await configured_service.register_file(
            "tenant",
            str(first),
            filename="first.txt",
            size=first.stat().st_size,
            mime_type="text/plain",
        )
        assert configured_service.queue.full() is True

        upload = UploadFile(filename="second.txt", file=BytesIO(b"bravo"))
        upload.content_type = "text/plain"
        limits = UploadLimits(max_upload_mb=10)

        with pytest.raises(HTTPException) as exc:
            await upload_file(
                file=[upload],
                files=None,
                limits=limits,
                data_dir=tmp_path,
                _=object(),
                tenant="tenant",
                ingest_service=configured_service,
            )
        expected_status = getattr(status, "HTTP_429_TOO_MANY_REQUESTS", 429)
        assert exc.value.status_code == expected_status
        assert exc.value.detail == "INGEST_QUEUE_FULL"
        tenant_dir = tmp_path / "tenant"
        assert tenant_dir.exists() is True
        assert not any(tenant_dir.iterdir())

    asyncio.run(scenario())


def test_ingest_endpoint_surfaces_429_when_queue_full(
    configured_service: IngestService, tmp_path: Path
) -> None:
    async def scenario() -> None:
        sample = _write_sample(tmp_path / "first.txt", "alpha")
        record, _ = await configured_service.register_file(
            "tenant",
            str(sample),
            filename="first.txt",
            size=sample.stat().st_size,
            mime_type="text/plain",
        )
        assert configured_service.queue.full() is True

        payload = IngestRequest(file_id=str(record.id), force=False)
        with Session(configured_service.engine) as session:
            with pytest.raises(HTTPException) as exc:
                await ingest_file(
                    payload,
                    object(),
                    configured_service,
                    session,
                    tenant="tenant",
                )
        expected_status = getattr(status, "HTTP_429_TOO_MANY_REQUESTS", 429)
        assert exc.value.status_code == expected_status
        assert exc.value.detail == "INGEST_QUEUE_FULL"

    asyncio.run(scenario())
