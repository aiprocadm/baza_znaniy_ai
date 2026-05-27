import asyncio
import hashlib
from decimal import Decimal
from datetime import timedelta
from pathlib import Path

import pytest
from sqlmodel import Session, select

from app.core.datetime_utils import utc_now
from app.ingest.service import IngestService, IngestWorker
from app.models import file as file_models
from app.models.entities import JobStatus, TenantRecord
from app.models.file import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    FileRecord,
    JobRecord,
    PageRecord,
)


@pytest.fixture
def sqlite_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "ingest.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DB_URL", db_url)
    file_models.get_engine.cache_clear()
    yield db_url
    file_models.get_engine.cache_clear()


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "document.txt"
    file_path.write_text("First page\nSecond page\nThird line", encoding="utf-8")
    return file_path


def test_register_creates_file_and_prevents_duplicates(sqlite_db: str, sample_file: Path) -> None:
    async def scenario() -> None:
        service = IngestService(max_retries=1, backoff_seconds=0)
        record, queued = await service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
            mime_type="text/plain",
        )
        assert queued is True
        assert record.status == file_models.FileStatus.QUEUED

        duplicate, again = await service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
            mime_type="text/plain",
        )
        assert again is False
        assert duplicate.id == record.id

        with Session(service.engine) as session:
            files = session.exec(select(FileRecord)).all()
            assert len(files) == 1
            assert files[0].status == file_models.FileStatus.QUEUED
            documents = session.exec(select(DocumentRecord)).all()
            assert len(documents) == 1
            assert documents[0].status == DocumentStatus.QUEUED
            assert documents[0].mime_type == "text/plain"
            jobs = session.exec(select(JobRecord)).all()
            assert len(jobs) == 1
            assert jobs[0].status == JobStatus.QUEUED
            assert jobs[0].attempt == 0

    asyncio.run(scenario())


def test_worker_processes_file_and_creates_pages(sqlite_db: str, sample_file: Path) -> None:
    async def scenario() -> None:
        service = IngestService(max_retries=1, backoff_seconds=0)
        worker = IngestWorker(service, embed_batch_size=2)

        record, queued = await service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
            mime_type="text/plain",
        )
        assert queued is True

        job = await service.queue.get()
        await worker._process(job)
        service.queue.task_done()

        with Session(service.engine) as session:
            file_obj = session.exec(select(FileRecord).where(FileRecord.id == record.id)).one()
            assert file_obj.status == file_models.FileStatus.COMPLETED
            assert file_obj.chunks > 0
            pages = session.exec(select(PageRecord).where(PageRecord.file_id == file_obj.id)).all()
            assert pages
            chunks = session.exec(
                select(ChunkRecord).where(ChunkRecord.page_id.in_([p.id for p in pages]))
            ).all()
            assert chunks
            assert all(page.tokens > 0 for page in pages)
            assert all(page.meta and page.meta.get("page") == page.number for page in pages)
            assert all(chunk.tokens > 0 for chunk in chunks)
            assert all(chunk.meta and "chunk" in chunk.meta for chunk in chunks)
            document = session.exec(
                select(DocumentRecord).where(DocumentRecord.id == file_obj.document_id)
            ).one()
            assert document.status == DocumentStatus.COMPLETED
            assert document.chunks == file_obj.chunks
            assert document.mime_type == "text/plain"
            jobs = session.exec(select(JobRecord).order_by(JobRecord.created_at)).all()
            assert jobs[-1].status == JobStatus.COMPLETED
            assert jobs[-1].payload and jobs[-1].payload.get("chunks") == file_obj.chunks

        # Re-register should not enqueue once completed
        _, queued_again = await service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
            mime_type="text/plain",
        )
        assert queued_again is False

    asyncio.run(scenario())


def test_worker_retries_failed_jobs(
    monkeypatch: pytest.MonkeyPatch, sqlite_db: str, sample_file: Path
) -> None:
    async def scenario() -> None:
        service = IngestService(max_retries=2, backoff_seconds=0)
        worker = IngestWorker(service)

        record, queued = await service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
            mime_type="text/plain",
        )
        assert queued is True

        async def failing_ingest(self, _job):
            raise RuntimeError("boom")

        monkeypatch.setattr(IngestWorker, "_ingest_file", failing_ingest)

        job = await service.queue.get()
        await worker._process(job)
        service.queue.task_done()

        requeued_job = await service.queue.get()
        service.queue.task_done()

        assert requeued_job.attempt == 1

        with Session(service.engine) as session:
            file_obj = session.exec(select(FileRecord).where(FileRecord.id == record.id)).one()
            assert file_obj.status == file_models.FileStatus.QUEUED
            assert file_obj.retries == 1
            assert file_obj.error == "boom"
            document = session.exec(
                select(DocumentRecord).where(DocumentRecord.id == file_obj.document_id)
            ).one()
            assert document.status == DocumentStatus.QUEUED
            jobs = session.exec(select(JobRecord).order_by(JobRecord.created_at)).all()
            assert len(jobs) == 2
            assert jobs[0].status == JobStatus.FAILED
            assert jobs[0].error == "boom"
            assert jobs[1].status == JobStatus.QUEUED
            assert jobs[1].attempt == 1

    asyncio.run(scenario())


def test_worker_finalization_handles_missing_file_record(
    monkeypatch: pytest.MonkeyPatch, sqlite_db: str, sample_file: Path
) -> None:
    async def scenario() -> None:
        service = IngestService(max_retries=0, backoff_seconds=0)
        worker = IngestWorker(service)

        record, queued = await service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
            mime_type="text/plain",
        )
        assert queued is True

        async def ingest_and_remove_file(self, _job):
            with Session(self.service.engine) as session:
                file_obj = session.get(FileRecord, record.id)
                assert file_obj is not None
                session.delete(file_obj)
                session.commit()
            return 1

        monkeypatch.setattr(IngestWorker, "_ingest_file", ingest_and_remove_file)

        job = await service.queue.get()
        assert job.job_record_id is not None
        await worker._process(job)
        service.queue.task_done()

        with Session(service.engine) as session:
            job_record = session.get(JobRecord, job.job_record_id)
            assert job_record is not None
            assert job_record.status == JobStatus.FAILED
            assert job_record.error == "FILE_MISSING"

    asyncio.run(scenario())


def test_hash_is_based_on_contents(sqlite_db: str, tmp_path: Path) -> None:
    async def scenario() -> None:
        first = tmp_path / "a.txt"
        second = tmp_path / "b.txt"
        first.write_text("hello", encoding="utf-8")
        second.write_text("hello", encoding="utf-8")

        service = IngestService(max_retries=0, backoff_seconds=0)

        record_one, queued_one = await service.register_file(
            "tenant",
            str(first),
            filename=first.name,
            size=first.stat().st_size,
            mime_type="text/plain",
        )
        record_two, queued_two = await service.register_file(
            "tenant",
            str(second),
            filename=second.name,
            size=second.stat().st_size,
            mime_type="text/plain",
        )

        assert queued_one is True
        assert queued_two is False
        assert record_two.id == record_one.id

        digest = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        with Session(service.engine) as session:
            file_obj = session.exec(select(FileRecord).where(FileRecord.sha256 == digest)).one()
            assert Path(file_obj.path).name == Path(first).name

    asyncio.run(scenario())


def test_cross_process_worker_flow(sqlite_db: str, sample_file: Path) -> None:
    async def scenario() -> None:
        api_service = IngestService(
            max_retries=1, backoff_seconds=0, auto_process=False, use_local_queue=False
        )
        record, queued = await api_service.register_file(
            "tenant",
            str(sample_file),
            filename=sample_file.name,
            size=sample_file.stat().st_size,
            mime_type="text/plain",
        )
        assert queued is True

        worker_service = IngestService(
            max_retries=1, backoff_seconds=0, auto_process=False, use_local_queue=False
        )
        worker = IngestWorker(worker_service, embed_batch_size=2)

        job = worker_service.dequeue_next_job()
        assert job is not None
        assert job.job_record_id is not None

        await worker._process(job)

        with Session(worker_service.engine) as session:
            file_obj = session.exec(select(FileRecord).where(FileRecord.id == record.id)).one()
            assert file_obj.status == file_models.FileStatus.COMPLETED
            jobs = session.exec(select(JobRecord).order_by(JobRecord.created_at)).all()
            assert jobs[-1].status == JobStatus.COMPLETED

    asyncio.run(scenario())


def test_service_uses_settings_for_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_MAX_RETRIES", "5")
    monkeypatch.setenv("INGEST_BACKOFF_SECONDS", "2.5")
    monkeypatch.setenv("INGEST_QUEUE_SIZE", "0")
    monkeypatch.setenv("INGEST_PROCESSING_TIMEOUT_SECONDS", "7")
    from app.core import config as config_module

    config_module.get_settings.cache_clear()
    # tests/test_core_services.py can swap app.core.config in sys.modules;
    # ingest.service still holds the original `from X import f` binding,
    # so we must clear that one too — otherwise it serves a stale cache.
    from app.ingest import service as ingest_service_module

    ingest_service_module.get_settings.cache_clear()

    service = IngestService()

    assert service.max_retries == 5
    assert service.backoff_seconds == 2.5
    assert service.queue_maxsize == 0
    assert service.queue.maxsize == 0
    assert service.processing_timeout_seconds == 7.0

    config_module.get_settings.cache_clear()


def test_service_accepts_unbounded_queue_size_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST_QUEUE_SIZE", "unbounded")
    from app.core import config as config_module

    config_module.get_settings.cache_clear()

    service = IngestService()

    assert service.queue_maxsize == 0
    assert service.queue.maxsize == 0

    config_module.get_settings.cache_clear()


def test_service_accepts_unbounded_queue_size_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST_QUEUE_SIZE", "4")
    from app.core import config as config_module

    config_module.get_settings.cache_clear()

    service = IngestService(queue_maxsize="infinite")

    assert service.queue_maxsize == 0
    assert service.queue.maxsize == 0

    config_module.get_settings.cache_clear()


def test_service_accepts_infinite_queue_size_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST_QUEUE_SIZE", "4")
    from app.core import config as config_module

    config_module.get_settings.cache_clear()

    service = IngestService(queue_maxsize=float("inf"))

    assert service.queue_maxsize == 0
    assert service.queue.maxsize == 0

    config_module.get_settings.cache_clear()


def test_service_accepts_decimal_infinite_queue_size_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGEST_QUEUE_SIZE", "4")
    from app.core import config as config_module

    config_module.get_settings.cache_clear()

    service = IngestService(queue_maxsize=Decimal("Infinity"))

    assert service.queue_maxsize == 0
    assert service.queue.maxsize == 0

    config_module.get_settings.cache_clear()


def test_settings_accept_decimal_infinite_queue_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_QUEUE_SIZE", raising=False)
    from app.core.config import Settings

    settings = Settings(ingest_queue_size=Decimal("Infinity"))

    assert settings.ingest_queue_size == 0


def test_service_rejects_queue_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_QUEUE_SIZE", "1")
    from app.core import config as config_module

    config_module.get_settings.cache_clear()

    loop = asyncio.new_event_loop()
    try:
        try:
            previous_loop = asyncio.get_running_loop()
        except RuntimeError:
            previous_loop = None
        asyncio.set_event_loop(loop)
        queue = asyncio.Queue(maxsize=1)
        with pytest.raises(ValueError):
            IngestService(queue=queue, queue_maxsize=2)
    finally:
        asyncio.set_event_loop(previous_loop)
        loop.close()
        config_module.get_settings.cache_clear()


def test_perform_maintenance_resets_stale_processing(
    monkeypatch: pytest.MonkeyPatch, sqlite_db: str
) -> None:
    monkeypatch.setenv("INGEST_JOB_RETENTION_DAYS", "1")
    from app.core import config as config_module

    config_module.get_settings.cache_clear()

    service = IngestService(max_retries=0, backoff_seconds=0)
    service.job_retention_days = 1

    stale_timestamp = utc_now() - timedelta(days=2)
    recent_timestamp = utc_now()
    stale_naive = stale_timestamp.replace(tzinfo=None)
    recent_naive = recent_timestamp.replace(tzinfo=None)

    with Session(service.engine) as session:
        tenant = TenantRecord(tenant_id="tenant", slug="tenant")
        session.add(tenant)
        session.commit()

        document = DocumentRecord(
            tenant_id="tenant",
            tenant_slug="tenant",
            sha256="sha",
            mime_type="text/plain",
            status=DocumentStatus.PROCESSING,
            error="processing",
            chunks=None,
            created_at=stale_naive,
            updated_at=stale_naive,
        )
        session.add(document)
        session.flush()

        file_obj = FileRecord(
            tenant_id="tenant",
            sha256="sha",
            document_id=document.id,
            path="/tmp/file.txt",
            filename="file.txt",
            size=128,
            status=file_models.FileStatus.PROCESSING,
            retries=1,
            error="processing",
            chunks=None,
            created_at=stale_naive,
            updated_at=stale_naive,
        )
        session.add(file_obj)
        session.flush()

        stale_job = JobRecord(
            tenant_id="tenant",
            tenant_slug="tenant",
            job_type="ingest",
            status=JobStatus.COMPLETED,
            priority=0,
            error=None,
            resource_id=str(file_obj.id),
            attempt=0,
            payload={"kind": "stale"},
            created_at=stale_naive,
            updated_at=stale_naive,
            started_at=stale_naive,
            finished_at=stale_naive,
        )

        recent_job = JobRecord(
            tenant_id="tenant",
            tenant_slug="tenant",
            job_type="ingest",
            status=JobStatus.PROCESSING,
            priority=0,
            error=None,
            resource_id=str(file_obj.id),
            attempt=0,
            payload={"kind": "recent"},
            created_at=recent_naive,
            updated_at=recent_naive,
            started_at=recent_naive,
            finished_at=None,
        )

        session.add(stale_job)
        session.add(recent_job)
        session.commit()

        stale_job_id = stale_job.id
        recent_job_id = recent_job.id
        file_id = file_obj.id
        document_id = document.id

    try:
        service._perform_maintenance()

        with Session(service.engine) as session:
            assert session.get(JobRecord, stale_job_id) is None
            assert session.get(JobRecord, recent_job_id) is not None

            refreshed_file = session.get(FileRecord, file_id)
            assert refreshed_file is not None
            assert refreshed_file.status == file_models.FileStatus.FAILED
            assert refreshed_file.error == "STALE_PROCESSING"
            assert refreshed_file.updated_at > stale_naive

            refreshed_document = session.get(DocumentRecord, document_id)
            assert refreshed_document is not None
            assert refreshed_document.status == DocumentStatus.FAILED
            assert refreshed_document.error == "STALE_PROCESSING"
            assert refreshed_document.updated_at > stale_naive
    finally:
        config_module.get_settings.cache_clear()


def test_dequeue_recovers_stuck_processing_job(
    monkeypatch: pytest.MonkeyPatch, sqlite_db: str, sample_file: Path
) -> None:
    async def scenario() -> None:
        monkeypatch.setenv("INGEST_PROCESSING_TIMEOUT_SECONDS", "5")
        from app.core import config as config_module

        config_module.get_settings.cache_clear()
        try:
            service = IngestService(
                max_retries=1, backoff_seconds=0, auto_process=False, use_local_queue=False
            )
            record, queued = await service.register_file(
                "tenant",
                str(sample_file),
                filename=sample_file.name,
                size=sample_file.stat().st_size,
                mime_type="text/plain",
            )
            assert queued is True

            stale_started = utc_now() - timedelta(seconds=30)
            stale_naive = stale_started.replace(tzinfo=None)
            with Session(service.engine) as session:
                job_record = session.exec(
                    select(JobRecord).where(JobRecord.resource_id == str(record.id))
                ).one()
                job_record.status = JobStatus.PROCESSING
                job_record.started_at = stale_naive
                job_record.updated_at = stale_naive
                job_record.attempt = 0
                job_record.error = "WORKER_CRASHED"
                job_record.payload = {
                    **dict(job_record.payload or {}),
                    "attempt": 0,
                }
                session.add(job_record)
                session.commit()
                job_id = job_record.id

            recovered_job = service.dequeue_next_job()
            assert recovered_job is not None
            assert recovered_job.job_record_id == job_id
            assert recovered_job.attempt == 1

            with Session(service.engine) as session:
                refreshed = session.get(JobRecord, job_id)
                assert refreshed is not None
                assert refreshed.status == JobStatus.PROCESSING
                assert refreshed.attempt == 1
                assert refreshed.error == "RECOVERED_STUCK_JOB"
                assert refreshed.payload is not None
                assert refreshed.payload.get("recovered_stuck_job") is True
                assert refreshed.payload.get("recovery_reason") == "RECOVERED_STUCK_JOB"
                assert refreshed.payload.get("attempt") == 1
        finally:
            config_module.get_settings.cache_clear()

    asyncio.run(scenario())
