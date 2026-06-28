import asyncio
import hashlib
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlmodel import Session, select

from app.core.datetime_utils import utc_now
from app.ingest.service import (
    IngestJob,
    IngestService,
    IngestWorker,
    _extract_npa_fields,
    _parse_act_date,
)
from app.models import file as file_models
from app.models.entities import JobStatus, TenantRecord
from app.models.file import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    FileRecord,
    FileStatus,
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


# --- Characterization tests for dequeue_next_job() seams ---------------------
# These pin the behaviour of the two pure blocks extracted out of the
# ~157-line dequeue_next_job() so the split is provably behaviour-preserving.


def test_extract_row_id_and_status_from_mapping() -> None:
    row = SimpleNamespace(_mapping={"id": 7, "status": "queued"})
    assert IngestService._extract_row_id_and_status(row) == (7, "queued")


def test_extract_row_id_and_status_from_indexable() -> None:
    # A plain tuple row (id, status) — second access path.
    assert IngestService._extract_row_id_and_status((9, "processing")) == (9, "processing")


def test_extract_row_id_and_status_from_attributes() -> None:
    row = SimpleNamespace(id=3, status="queued")
    assert IngestService._extract_row_id_and_status(row) == (3, "queued")


def test_ingest_job_from_record_parses_payload() -> None:
    record = JobRecord(
        tenant_id="acme",
        payload={
            "file_id": "42",
            "filename": "doc.pdf",
            "path": "/data/doc.pdf",
            "sha256": "abc",
            "document_id": "5",
            "attempt": "2",
        },
    )
    record.id = 11

    job = IngestService._ingest_job_from_record(record)

    assert isinstance(job, IngestJob)
    assert job.file_id == 42
    assert job.document_id == 5
    assert job.attempt == 2
    assert job.tenant_id == "acme"
    assert job.filename == "doc.pdf"
    assert job.job_record_id == 11


def test_ingest_job_from_record_returns_none_without_file_id() -> None:
    record = JobRecord(tenant_id="acme", payload={"filename": "x"})
    assert IngestService._ingest_job_from_record(record) is None


def test_ingest_job_from_record_falls_back_to_record_attempt() -> None:
    record = JobRecord(tenant_id="acme", attempt=4, payload={"file_id": 1})
    job = IngestService._ingest_job_from_record(record)
    assert job is not None
    assert job.attempt == 4


# --- Characterization tests for IngestService.__init__ resolver seams --------
# Pin the param -> env -> settings precedence lifted out of the ~109-line
# __init__ so the split is provably behaviour-preserving.


def test_resolve_retries_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(ingest_max_retries=3)

    # Explicit param wins over everything.
    monkeypatch.setenv("INGEST_MAX_RETRIES", "9")
    assert IngestService._resolve_retries(5, settings) == 5

    # Env wins when no param.
    assert IngestService._resolve_retries(None, settings) == 9

    # Invalid env falls back to settings.
    monkeypatch.setenv("INGEST_MAX_RETRIES", "not-an-int")
    assert IngestService._resolve_retries(None, settings) == 3

    # Settings when neither param nor env.
    monkeypatch.delenv("INGEST_MAX_RETRIES", raising=False)
    assert IngestService._resolve_retries(None, settings) == 3


def test_resolve_backoff_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SimpleNamespace(ingest_backoff_seconds=1.5)
    monkeypatch.delenv("INGEST_BACKOFF_SECONDS", raising=False)
    monkeypatch.delenv("INGEST_BACKOFF_BASE", raising=False)

    assert IngestService._resolve_backoff(0.25, settings) == 0.25

    monkeypatch.setenv("INGEST_BACKOFF_BASE", "2.0")
    assert IngestService._resolve_backoff(None, settings) == 2.0

    monkeypatch.setenv("INGEST_BACKOFF_SECONDS", "bad")
    assert IngestService._resolve_backoff(None, settings) == 1.5

    monkeypatch.delenv("INGEST_BACKOFF_SECONDS", raising=False)
    monkeypatch.delenv("INGEST_BACKOFF_BASE", raising=False)
    assert IngestService._resolve_backoff(None, settings) == 1.5


# --- Characterization tests for the NPA metadata seams -----------------------
# _parse_act_date was lifted out of the ~120-line _ingest_file so the lenient
# "dd.mm.yyyy or None" parsing is provable without a full ingest round-trip.
# _extract_npa_fields had no direct coverage at all; pin its precedence here.


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("01.02.2024", datetime(2024, 2, 1)),
        ("31.12.1999", datetime(1999, 12, 31)),
        ("2024-02-01", None),  # ISO format is not the expected dd.mm.yyyy
        ("32.13.2024", None),  # out-of-range day/month
        ("not a date", None),
        ("", None),
        (None, None),
        (20240201, None),  # non-string inputs degrade rather than raise
    ],
)
def test_parse_act_date_lenient(raw: object, expected: datetime | None) -> None:
    assert _parse_act_date(raw) == expected


def test_extract_npa_fields_metadata_fallback_wins_over_regex() -> None:
    content = "Тип акта: приказ\nномер: А-1"
    fields = _extract_npa_fields(
        content,
        metadata={"act_type": "федеральный закон", "reg_number": "Z-9"},
    )
    # When metadata supplies a fallback key it short-circuits the regex scan.
    assert fields["act_type"] == "федеральный закон"
    assert fields["reg_number"] == "Z-9"


def test_extract_npa_fields_regex_extraction_when_no_metadata() -> None:
    content = (
        "Тип акта: постановление\n"
        "Издатель: Минюст\n"
        "Дата принятия: 05.06.2021\n"
        "Дата вступления в силу: 10.06.2021\n"
    )
    fields = _extract_npa_fields(content)
    assert fields["act_type"] == "постановление"
    assert fields["issuer"] == "Минюст"
    assert fields["adoption_date"] == "05.06.2021"
    assert fields["effective_date"] == "10.06.2021"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, True),  # absent flag defaults to active
        ("да", True),
        ("true", True),
        ("1", True),
        ("нет", False),
        ("0", False),
        ("", False),
    ],
)
def test_extract_npa_fields_is_active_truthiness(raw: object, expected: bool) -> None:
    metadata = {} if raw is None else {"is_active": raw}
    assert _extract_npa_fields("", metadata=metadata)["is_active"] is expected


def test_extract_npa_fields_defaults_to_none_when_absent() -> None:
    fields = _extract_npa_fields("no structured metadata here")
    for key in ("act_type", "issuer", "reg_number", "adoption_date", "effective_date", "revision"):
        assert fields[key] is None


def test_apply_npa_metadata_writes_document_and_returns_attrs() -> None:
    # _apply_npa_metadata was lifted out of _ingest_file. Pin that it mutates the
    # document in place (content + parsed NPA fields) and returns the attrs for
    # chunk-meta reuse — all without touching a DB session.
    document = DocumentRecord(
        tenant_id="acme",
        sha256="sha",
        meta={"act_type": "приказ", "reg_number": "Z-9", "is_active": "нет"},
    )
    content = "Дата принятия: 05.06.2021\nДата вступления в силу: 10.06.2021\n"

    attrs = IngestWorker._apply_npa_metadata(document, content)

    assert attrs["act_type"] == "приказ"
    assert document.content == content
    assert document.act_type == "приказ"
    assert document.reg_number == "Z-9"
    assert document.is_active is False
    # dd.mm.yyyy strings are coerced to datetimes on the document.
    assert document.adoption_date == datetime(2021, 6, 5)
    assert document.effective_date == datetime(2021, 6, 10)


def test_apply_npa_metadata_defaults_active_and_clears_absent_fields() -> None:
    document = DocumentRecord(tenant_id="acme", sha256="sha")

    attrs = IngestWorker._apply_npa_metadata(document, "no structured metadata")

    assert attrs["is_active"] is True
    assert document.is_active is True
    assert document.act_type is None
    assert document.adoption_date is None
    assert document.effective_date is None


# --- Characterization tests for the _process FILE_MISSING seams --------------
# _load_file_and_job and _fail_job_file_missing were lifted out of two
# near-identical inline blocks in IngestWorker._process. Pin their behaviour
# here with a recording stand-in so no real DB engine is needed.


class _RecordingSession:
    """Minimal SQLModel-Session stand-in for the FILE_MISSING helpers."""

    def __init__(self, rows: dict | None = None) -> None:
        self._rows = rows or {}
        self.added: list = []
        self.commits = 0
        self.get_calls: list = []

    def get(self, model: object, ident: object) -> object:
        self.get_calls.append((model, ident))
        return self._rows.get((model, ident))

    def add(self, obj: object) -> None:
        self.added.append(obj)

    def commit(self) -> None:
        self.commits += 1


def test_fail_job_file_missing_marks_failed_and_commits() -> None:
    session = _RecordingSession()
    record = JobRecord(tenant_id="acme", status=JobStatus.PROCESSING, payload={"file_id": 1})

    IngestWorker._fail_job_file_missing(session, record)

    assert record.status == JobStatus.FAILED
    assert record.error == "FILE_MISSING"
    assert record.finished_at is not None
    assert record.finished_at == record.updated_at
    assert session.added == [record]
    assert session.commits == 1


def test_fail_job_file_missing_is_noop_without_record() -> None:
    session = _RecordingSession()

    IngestWorker._fail_job_file_missing(session, None)

    assert session.added == []
    assert session.commits == 0


def test_load_file_and_job_skips_job_lookup_without_record_id() -> None:
    file_rec = FileRecord(tenant_id="acme", sha256="x", path="p", filename="f", size=1)
    session = _RecordingSession({(FileRecord, 7): file_rec})
    job = IngestJob(
        tenant_id="acme",
        path="p",
        sha256="x",
        file_id=7,
        filename="f",
        document_id=None,
        job_record_id=None,
    )

    file_obj, job_record = IngestWorker._load_file_and_job(session, job)

    assert file_obj is file_rec
    assert job_record is None
    # JobRecord must never be queried when the job carries no record id.
    assert all(model is not JobRecord for model, _ in session.get_calls)


def test_load_file_and_job_fetches_both_when_record_id_present() -> None:
    file_rec = FileRecord(tenant_id="acme", sha256="x", path="p", filename="f", size=1)
    job_rec = JobRecord(tenant_id="acme", payload={"file_id": 7})
    session = _RecordingSession({(FileRecord, 7): file_rec, (JobRecord, 3): job_rec})
    job = IngestJob(
        tenant_id="acme",
        path="p",
        sha256="x",
        file_id=7,
        filename="f",
        document_id=None,
        job_record_id=3,
    )

    file_obj, job_record = IngestWorker._load_file_and_job(session, job)

    assert file_obj is file_rec
    assert job_record is job_rec


# --- Characterization tests for the _process finalization seam ---------------
# _finalize_job collapses _process's symmetric success/failure branches. Pin
# the per-row writes (status, error, chunks, retries, payload) for both
# outcomes without a real DB engine.


def test_finalize_job_success_writes_completed_and_clears_errors() -> None:
    session = _RecordingSession()
    file_obj = FileRecord(
        tenant_id="acme", sha256="x", path="p", filename="f", size=1, retries=2, error="old"
    )
    document = DocumentRecord(tenant_id="acme", sha256="x", error="old")
    job_record = JobRecord(tenant_id="acme", payload={"file_id": 1})

    IngestWorker._finalize_job(
        session,
        file_obj=file_obj,
        document=document,
        job_record=job_record,
        success=True,
        chunk_count=5,
        error_message=None,
        attempt=0,
    )

    assert file_obj.status == FileStatus.COMPLETED
    assert file_obj.error is None
    assert file_obj.chunks == 5
    assert file_obj.retries == 2  # unchanged on success
    assert document.status == DocumentStatus.COMPLETED
    assert document.error is None and document.chunks == 5
    assert job_record.status == JobStatus.COMPLETED
    assert job_record.error is None
    assert job_record.payload == {"file_id": 1, "chunks": 5, "attempt": 0}
    assert session.commits == 1


def test_finalize_job_failure_sets_retries_and_propagates_error() -> None:
    session = _RecordingSession()
    file_obj = FileRecord(tenant_id="acme", sha256="x", path="p", filename="f", size=1)
    document = DocumentRecord(tenant_id="acme", sha256="x")
    job_record = JobRecord(tenant_id="acme", payload={"file_id": 1})

    IngestWorker._finalize_job(
        session,
        file_obj=file_obj,
        document=document,
        job_record=job_record,
        success=False,
        chunk_count=0,
        error_message="boom",
        attempt=2,
    )

    assert file_obj.status == FileStatus.FAILED
    assert file_obj.retries == 3  # attempt + 1
    assert file_obj.error == "boom"
    assert file_obj.chunks == 0
    assert document.status == DocumentStatus.FAILED
    assert document.error == "boom"
    assert job_record.status == JobStatus.FAILED
    assert job_record.error == "boom"
    assert job_record.payload == {"file_id": 1, "chunks": 0, "attempt": 2}
    assert session.commits == 1


def test_finalize_job_handles_missing_document_and_job_record() -> None:
    session = _RecordingSession()
    file_obj = FileRecord(tenant_id="acme", sha256="x", path="p", filename="f", size=1)

    IngestWorker._finalize_job(
        session,
        file_obj=file_obj,
        document=None,
        job_record=None,
        success=True,
        chunk_count=3,
        error_message=None,
        attempt=0,
    )

    assert file_obj.status == FileStatus.COMPLETED
    assert file_obj.chunks == 3
    assert session.added == [file_obj]
    assert session.commits == 1


class _NullSession:
    """Stand-in Session for _begin_processing: every DB op is a no-op."""

    def __enter__(self) -> "_NullSession":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def get(self, *_: object) -> None:
        return None

    def add(self, *_: object) -> None:
        return None

    def commit(self) -> None:
        return None

    def scalars(self, *_: object) -> SimpleNamespace:
        # _begin_processing's page-cleanup select yields no rows.
        return SimpleNamespace(all=lambda: [])

    def execute(self, *_: object) -> None:
        return None


def test_process_propagates_cancellation_without_swallowing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A BaseException from _ingest_file must propagate, skipping finalization.

    Regression guard: finalization used to live in a ``finally`` whose ``return``
    (the FILE_MISSING path) silently swallowed in-flight ``CancelledError``,
    breaking task shutdown and leaving the job mismarked. The fix re-raises so
    stale-recovery can reclaim the still-PROCESSING job.
    """

    service = IngestService()
    service._engine = SimpleNamespace()  # sentinel; Session is patched out below
    worker = IngestWorker(service)

    file_obj = SimpleNamespace(
        id=1,
        status=FileStatus.QUEUED,
        error=None,
        chunks=None,
        updated_at=None,
        document_id=None,
    )
    job_record = SimpleNamespace(status=JobStatus.QUEUED, started_at=None, updated_at=None)

    monkeypatch.setattr("app.ingest.service.Session", lambda *a, **k: _NullSession())
    # Pre-ingest setup runs `select(...).where(...)`; stub it so the page-cleanup
    # query yields no rows regardless of the SQLModel backend on this machine.
    monkeypatch.setattr(
        "app.ingest.service.select",
        lambda *a, **k: SimpleNamespace(where=lambda *a, **k: None),
    )

    load_calls = {"n": 0}

    def fake_load(session: object, job: object) -> tuple:
        load_calls["n"] += 1
        # First call is the pre-ingest setup (file present, proceed). Any later
        # call is the finalization phase, which the fix must never reach here.
        return (file_obj, job_record) if load_calls["n"] == 1 else (None, job_record)

    monkeypatch.setattr(IngestWorker, "_load_file_and_job", staticmethod(fake_load))

    fail_calls: list = []
    monkeypatch.setattr(
        IngestWorker,
        "_fail_job_file_missing",
        staticmethod(lambda *a, **k: fail_calls.append(1)),
    )
    finalize_calls: list = []
    monkeypatch.setattr(
        IngestWorker,
        "_finalize_job",
        staticmethod(lambda *a, **k: finalize_calls.append(1)),
    )

    async def _boom(self: object, job: object) -> int:
        raise asyncio.CancelledError()

    monkeypatch.setattr(IngestWorker, "_ingest_file", _boom)

    handled: list = []

    async def _handle(self: object, job: object) -> None:
        handled.append(1)

    monkeypatch.setattr(IngestWorker, "_handle_failure", _handle)

    job = IngestJob(
        tenant_id="acme",
        path="p",
        sha256="s",
        file_id=1,
        filename="f",
        document_id=None,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(worker._process(job))

    assert load_calls["n"] == 1  # finalization phase never entered
    assert finalize_calls == []
    assert fail_calls == []  # the swallowing FILE_MISSING return was never reached
    assert handled == []


# --- Orchestration tests for _process over its extracted phases --------------
# With _begin_processing/_finalize as seams, _process's control flow is testable
# by stubbing the phases directly - no Session, select or DB engine involved.


def _make_worker() -> IngestWorker:
    service = IngestService()
    service._engine = SimpleNamespace()  # never touched; phases are stubbed
    return IngestWorker(service)


def _job() -> IngestJob:
    return IngestJob(
        tenant_id="acme", path="p", sha256="s", file_id=1, filename="f", document_id=None
    )


def test_process_skips_ingest_when_begin_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _make_worker()
    monkeypatch.setattr(IngestWorker, "_begin_processing", lambda self, job: False)

    ingest_calls: list = []

    async def _ingest(self: object, job: object) -> int:
        ingest_calls.append(1)
        return 0

    monkeypatch.setattr(IngestWorker, "_ingest_file", _ingest)
    finalize_calls: list = []
    monkeypatch.setattr(IngestWorker, "_finalize", lambda self, job, **k: finalize_calls.append(1))

    asyncio.run(worker._process(_job()))

    assert ingest_calls == []  # already-handled job is not re-ingested
    assert finalize_calls == []


def test_process_finalizes_success_and_skips_handle_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _make_worker()
    monkeypatch.setattr(IngestWorker, "_begin_processing", lambda self, job: True)

    async def _ingest(self: object, job: object) -> int:
        return 7

    monkeypatch.setattr(IngestWorker, "_ingest_file", _ingest)

    captured: dict = {}

    def _finalize(self: object, job: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(IngestWorker, "_finalize", _finalize)

    handled: list = []

    async def _handle(self: object, job: object) -> None:
        handled.append(1)

    monkeypatch.setattr(IngestWorker, "_handle_failure", _handle)

    asyncio.run(worker._process(_job()))

    assert captured == {"success": True, "chunk_count": 7, "error_message": None}
    assert handled == []


def test_process_finalizes_failure_and_runs_handle_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _make_worker()
    monkeypatch.setattr(IngestWorker, "_begin_processing", lambda self, job: True)

    async def _ingest(self: object, job: object) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(IngestWorker, "_ingest_file", _ingest)

    captured: dict = {}

    def _finalize(self: object, job: object, **kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(IngestWorker, "_finalize", _finalize)

    handled: list = []

    async def _handle(self: object, job: object) -> None:
        handled.append(1)

    monkeypatch.setattr(IngestWorker, "_handle_failure", _handle)

    asyncio.run(worker._process(_job()))

    assert captured["success"] is False
    assert captured["error_message"] == "boom"
    assert captured["chunk_count"] == 0  # nothing ingested before the failure
    assert handled == [1]


# --- Characterization tests for the _ingest_file pure seams ------------------
# Token counting (with its silent char-length fallback) and the page/chunk sha
# formulas were lifted out of the ~115-line _ingest_file. Pin them here.


def test_count_tokens_uses_encoder_length() -> None:
    encoder = SimpleNamespace(encode=lambda text: list(text))
    assert IngestWorker._count_tokens(encoder, "abcd") == 4


def test_count_tokens_falls_back_to_char_length_on_error() -> None:
    def _raise(_text: str) -> list:
        raise RuntimeError("tokenizer down")

    encoder = SimpleNamespace(encode=_raise)
    assert IngestWorker._count_tokens(encoder, "hello") == 5


def test_page_sha_matches_documented_format() -> None:
    expected = hashlib.sha256("doc:3:5".encode("utf-8")).hexdigest()
    assert IngestWorker._page_sha("doc", 3, "abcde") == expected


def test_chunk_sha_matches_documented_format() -> None:
    expected = hashlib.sha256("doc:3:2:body".encode("utf-8")).hexdigest()
    assert IngestWorker._chunk_sha("doc", 3, 2, "body") == expected


def test_page_sha_keys_on_length_while_chunk_sha_keys_on_text() -> None:
    a = IngestWorker._page_sha("doc", 1, "text")
    assert a == IngestWorker._page_sha("doc", 1, "text")  # deterministic
    assert a != IngestWorker._page_sha("doc", 2, "text")  # page-number sensitive
    # page sha keys on len(text), so equal-length texts collide by design...
    assert IngestWorker._page_sha("doc", 1, "ab") == IngestWorker._page_sha("doc", 1, "cd")
    # ...whereas chunk sha keys on the text itself.
    assert IngestWorker._chunk_sha("doc", 1, 1, "ab") != IngestWorker._chunk_sha("doc", 1, 1, "cd")


def test_chunk_meta_merges_provenance_and_attrs() -> None:
    parse_result = SimpleNamespace(
        parser_backend_used="pdfminer", fallback_reason=None, ocr_used=False
    )
    meta = IngestWorker._chunk_meta(
        "docsha", 2, 3, parse_result, {"act_type": "law", "issuer": "X"}
    )
    assert meta["document_sha"] == "docsha"
    assert meta["page"] == 2
    assert meta["chunk"] == 3
    assert meta["parser_backend"] == "pdfminer"
    assert meta["fallback_reason"] is None
    assert meta["ocr_used"] is False
    assert meta["act_type"] == "law"
    assert meta["issuer"] == "X"


def test_chunk_meta_attrs_override_provenance_on_key_clash() -> None:
    parse_result = SimpleNamespace(parser_backend_used="b", fallback_reason=None, ocr_used=False)
    # `**attrs` is spread last, so a clashing key (here "page") wins by design.
    meta = IngestWorker._chunk_meta("doc", 1, 1, parse_result, {"page": 99})
    assert meta["page"] == 99


def test_build_chunk_payload_shape() -> None:
    payload = IngestWorker._build_chunk_payload(
        filename="f.pdf",
        page_number=4,
        chunk_sha="abc",
        chunk_text="body",
        chunk_tokens=5,
        meta={"k": "v"},
        tenant_id="acme",
    )
    assert payload == {
        "file": "f.pdf",
        "page": 4,
        "sha256": "abc",
        "text": "body",
        "tokens": 5,
        "meta": {"k": "v"},
        "owner": "acme",
        "tenant_id": "acme",
    }


def test_build_chunk_payload_defaults_empty_meta() -> None:
    payload = IngestWorker._build_chunk_payload(
        filename="f",
        page_number=1,
        chunk_sha="s",
        chunk_text="t",
        chunk_tokens=1,
        meta=None,
        tenant_id="acme",
    )
    assert payload["meta"] == {}
