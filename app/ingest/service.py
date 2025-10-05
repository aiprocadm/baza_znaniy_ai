"""Async ingestion queue and worker implementation."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from asyncio import QueueEmpty, QueueFull
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from sqlmodel import Session, delete, select

from app.core.config import get_settings
from app.ingest.chunking import _chunk, _get_tokenizer, iter_document_pages
from app.models.entities import JobStatus
from app.models.file import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    FileRecord,
    FileStatus,
    JobRecord,
    PageRecord,
    get_engine,
)
from app.services import vectorstore


logger = logging.getLogger(__name__)


@dataclass
class IngestJob:
    """Descriptor for a queued ingestion task."""

    tenant_id: str
    path: str
    sha256: str
    file_id: int
    filename: str
    document_id: Optional[int]
    job_record_id: Optional[int] = None
    attempt: int = 0


class IngestQueueFullError(RuntimeError):
    """Raised when the ingestion queue cannot accept additional jobs."""


class IngestService:
    """Service responsible for computing file hashes and enqueueing jobs."""

    def __init__(
        self,
        *,
        queue: Optional[asyncio.Queue[Optional[IngestJob]]] = None,
        queue_maxsize: Optional[int] = None,
        max_retries: Optional[int] = None,
        backoff_seconds: Optional[float] = None,
        engine=None,
        auto_process: bool = False,
    ) -> None:
        settings = get_settings()
        retries = settings.ingest_max_retries if max_retries is None else max_retries
        backoff = (
            settings.ingest_backoff_seconds if backoff_seconds is None else backoff_seconds
        )
        queue_size = queue_maxsize or settings.ingest_queue_size
        self.queue_maxsize = max(1, int(queue_size))
        self.queue: asyncio.Queue[Optional[IngestJob]] = queue or asyncio.Queue(
            maxsize=self.queue_maxsize
        )
        self.max_retries = max(0, int(retries))
        self.backoff_seconds = max(0.0, float(backoff))
        self._engine = engine
        self.auto_process = auto_process
        self.worker: "IngestWorker" | None = None
        self._scheduler: AsyncIOScheduler | None = None
        self._worker_job_id: str | None = None
        self._maintenance_job_id: str | None = None
        self.worker_interval_seconds = max(
            0.1, float(settings.ingest_worker_interval_seconds)
        )
        self.maintenance_cron = settings.ingest_maintenance_cron
        self.job_retention_days = max(1, int(settings.ingest_job_retention_days))

    @property
    def engine(self):
        if self._engine is None:
            self._engine = get_engine()
        return self._engine

    def set_worker(self, worker: "IngestWorker") -> None:
        """Register the worker instance responsible for background processing."""

        self.worker = worker

    def configure_scheduler(self, scheduler: AsyncIOScheduler) -> None:
        """Attach an :class:`AsyncIOScheduler` used for background jobs."""

        self._scheduler = scheduler

    def _scheduler_job_name(self, suffix: str) -> str:
        return f"ingest-{id(self)}-{suffix}"

    def ensure_background_worker(self) -> None:
        """Start a background worker thread when auto-processing is enabled."""

        if not self.auto_process:
            return
        if self.worker is None:
            self.worker = IngestWorker(self)
        if self._scheduler is None:
            raise RuntimeError("Ingest scheduler has not been configured")
        if self._worker_job_id is None:
            trigger = IntervalTrigger(seconds=self.worker_interval_seconds)
            job = self._scheduler.add_job(
                self.worker.drain,
                trigger=trigger,
                id=self._scheduler_job_name("worker"),
                max_instances=1,
                coalesce=True,
            )
            self._worker_job_id = job.id
        if self.maintenance_cron and self._maintenance_job_id is None:
            try:
                trigger = CronTrigger.from_crontab(self.maintenance_cron)
            except ValueError:
                logger.error(
                    "Invalid ingest maintenance cron expression: %s",
                    self.maintenance_cron,
                )
            else:
                job = self._scheduler.add_job(
                    self.run_maintenance,
                    trigger=trigger,
                    id=self._scheduler_job_name("maintenance"),
                    max_instances=1,
                    coalesce=True,
                )
                self._maintenance_job_id = job.id

    async def stop_background_worker(self) -> None:
        """Signal the background worker thread to shut down."""

        if self.worker is not None:
            await self.worker.shutdown()
        if self._scheduler is None:
            return
        if self._worker_job_id is not None:
            try:
                self._scheduler.remove_job(self._worker_job_id)
            except JobLookupError:  # pragma: no cover - defensive cleanup
                pass
            self._worker_job_id = None
        if self._maintenance_job_id is not None:
            try:
                self._scheduler.remove_job(self._maintenance_job_id)
            except JobLookupError:  # pragma: no cover - defensive cleanup
                pass
            self._maintenance_job_id = None

    async def run_maintenance(self) -> None:
        """Execute periodic maintenance tasks for ingest metadata."""

        try:
            await asyncio.to_thread(self._perform_maintenance)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Ingest maintenance job failed")

    def _perform_maintenance(self) -> None:
        """Prune stale job records and reset stuck files."""

        cutoff = datetime.utcnow() - timedelta(days=self.job_retention_days)
        removed_jobs = 0
        reset_files = 0
        with Session(self.engine) as session:
            delete_result = session.exec(
                delete(JobRecord).where(
                    JobRecord.finished_at != None,  # noqa: E711 - SQLAlchemy comparison
                    JobRecord.finished_at < cutoff,
                )
            )
            try:
                removed_jobs = int(getattr(delete_result, "rowcount", 0) or 0)
            except Exception:  # pragma: no cover - defensive fallback
                removed_jobs = 0

            stale_files = session.exec(
                select(FileRecord).where(
                    FileRecord.status == FileStatus.PROCESSING,
                    FileRecord.updated_at < cutoff,
                )
            ).all()

            for file_obj in stale_files:
                file_obj.status = FileStatus.FAILED
                file_obj.error = "STALE_PROCESSING"
                file_obj.updated_at = datetime.utcnow()
                session.add(file_obj)
                if file_obj.document_id is not None:
                    document = session.get(DocumentRecord, file_obj.document_id)
                    if document:
                        document.status = DocumentStatus.FAILED
                        document.error = "STALE_PROCESSING"
                        document.updated_at = datetime.utcnow()
                        session.add(document)
            reset_files = len(stale_files)
            if removed_jobs or reset_files:
                session.commit()

        if removed_jobs or reset_files:
            logger.info(
                "Ingest maintenance removed %d jobs and reset %d files", removed_jobs, reset_files
            )

    @staticmethod
    def _hash_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _make_job(self, file_obj: FileRecord, *, attempt: int = 0) -> IngestJob:
        if file_obj.id is None:
            raise ValueError("FileRecord must be persisted before enqueueing")
        return IngestJob(
            tenant_id=file_obj.tenant_id,
            path=file_obj.path,
            sha256=file_obj.sha256,
            file_id=file_obj.id,
            filename=file_obj.filename,
            document_id=file_obj.document_id,
            attempt=attempt,
        )

    def _create_job_record(self, job: IngestJob, session: Session | None = None) -> JobRecord:
        payload = {
            "file_id": job.file_id,
            "sha256": job.sha256,
            "path": job.path,
            "filename": job.filename,
            "attempt": job.attempt,
        }
        if job.document_id is not None:
            payload["document_id"] = job.document_id

        manage_session = session is None
        session_obj = session or Session(self.engine)
        try:
            record = JobRecord(
                tenant_id=job.tenant_id,
                tenant_slug=job.tenant_id,
                job_type="ingest",
                status=JobStatus.QUEUED,
                priority=0,
                error=None,
                resource_id=str(job.file_id),
                attempt=job.attempt,
                payload=payload,
            )
            session_obj.add(record)
            session_obj.flush()
            session_obj.refresh(record)
            if manage_session:
                session_obj.commit()
            return record
        finally:
            if manage_session:
                session_obj.close()

    async def enqueue_job(
        self, file_obj: FileRecord, *, attempt: int = 0, session: Session | None = None
    ) -> IngestJob:
        if self.queue.full():
            raise IngestQueueFullError("INGEST_QUEUE_FULL")

        job = self._make_job(file_obj, attempt=attempt)
        manage_session = session is None
        session_obj = session or Session(self.engine)

        try:
            job_record = self._create_job_record(job, session=session_obj)
            job.job_record_id = job_record.id
            try:
                self.queue.put_nowait(job)
            except QueueFull as exc:
                raise IngestQueueFullError("INGEST_QUEUE_FULL") from exc
            if manage_session:
                session_obj.commit()
        except IngestQueueFullError:
            if manage_session:
                session_obj.rollback()
            raise
        finally:
            if manage_session:
                session_obj.close()

        worker = getattr(self, "worker", None)
        if worker is not None:
            worker.ensure_started()
        return job

    async def register_file(
        self,
        tenant_id: str,
        path: str,
        *,
        filename: str,
        size: int,
        mime_type: Optional[str] = None,
    ) -> Tuple[FileRecord, bool]:
        """Register a file upload and enqueue ingestion when necessary."""

        sha = self._hash_file(path)
        queued = False
        detected_mime = mime_type or "application/octet-stream"
        with Session(self.engine) as session:
            try:
                document = session.exec(
                    select(DocumentRecord).where(
                        DocumentRecord.tenant_id == tenant_id,
                        DocumentRecord.tenant_slug == tenant_id,
                        DocumentRecord.sha256 == sha,
                    )
                ).first()
                if document is None:
                    document = DocumentRecord(
                        tenant_id=tenant_id,
                        tenant_slug=tenant_id,
                        sha256=sha,
                        mime_type=detected_mime,
                        status=DocumentStatus.QUEUED,
                        error=None,
                        chunks=None,
                    )
                    session.add(document)
                else:
                    document.updated_at = datetime.utcnow()
                    if mime_type and document.mime_type != mime_type:
                        document.mime_type = mime_type
                    if document.status == DocumentStatus.FAILED:
                        document.status = DocumentStatus.QUEUED
                        document.error = None
                    document.chunks = None
                    session.add(document)

                session.flush()

                statement = select(FileRecord).where(
                    FileRecord.tenant_id == tenant_id, FileRecord.sha256 == sha
                )
                file_obj = session.exec(statement).first()
                if file_obj is None:
                    file_obj = FileRecord(
                        tenant_id=tenant_id,
                        sha256=sha,
                        document_id=document.id,
                        path=path,
                        filename=filename,
                        size=size,
                        status=FileStatus.QUEUED,
                        retries=0,
                        error=None,
                        chunks=None,
                    )
                    session.add(file_obj)
                    queued = True
                else:
                    file_obj.updated_at = datetime.utcnow()
                    file_obj.document_id = document.id
                    if file_obj.status == FileStatus.FAILED:
                        file_obj.path = path
                        file_obj.filename = filename
                        file_obj.size = size
                        file_obj.status = FileStatus.QUEUED
                        file_obj.retries = 0
                        file_obj.error = None
                        file_obj.chunks = None
                        queued = True
                    session.add(file_obj)

                session.flush()
                if queued:
                    await self.enqueue_job(file_obj, session=session)
                session.commit()
            except IngestQueueFullError:
                session.rollback()
                raise
            except Exception:
                session.rollback()
                raise
            else:
                session.refresh(file_obj)
                session.refresh(document)

        return file_obj, queued


class IngestWorker:
    """Worker that consumes the queue and stores parsing metadata."""

    def __init__(
        self,
        service: IngestService,
        *,
        embed_batch_size: Optional[int] = None,
    ) -> None:
        self.service = service
        settings = get_settings()
        default_batch = settings.embed_batch_size
        self.embed_batch_size = max(1, int(embed_batch_size or default_batch))
        self._tokenizer = _get_tokenizer()
        chunk = int(os.getenv("RAG_CHUNK", "900"))
        overlap = int(os.getenv("RAG_OVERLAP", "140"))
        self.chunk_size = chunk if chunk > 0 else 1
        self.overlap = overlap if overlap > 0 else 0
        if self.overlap >= self.chunk_size:
            self.overlap = max(0, self.chunk_size - 1)
        self._stop = False
        self.service.worker = self

    def ensure_started(self) -> None:
        if self.service.auto_process:
            self.service.ensure_background_worker()

    async def run(self) -> None:
        logger.debug("ingest worker run loop entered")
        while True:
            processed = await self.drain()
            if self._stop and self.service.queue.empty():
                break
            if processed == 0:
                await asyncio.sleep(0.1)

    async def drain(self, *, limit: int | None = None) -> int:
        """Process pending jobs without blocking for new items."""

        processed = 0
        while True:
            if limit is not None and processed >= limit:
                break
            try:
                job = self.service.queue.get_nowait()
            except QueueEmpty:
                break
            if job is None:
                self.service.queue.task_done()
                continue
            logger.debug("ingest worker processing job %s", job)
            try:
                await self._process(job)
            except Exception:  # pragma: no cover - defensive
                await self._handle_failure(job)
            finally:
                self.service.queue.task_done()
            processed += 1
        return processed

    async def _process(self, job: IngestJob) -> None:
        with Session(self.service.engine) as session:
            file_obj = session.get(FileRecord, job.file_id)
            job_record = (
                session.get(JobRecord, job.job_record_id)
                if job.job_record_id is not None
                else None
            )
            if not file_obj:
                if job_record:
                    job_record.status = JobStatus.FAILED
                    job_record.error = "FILE_MISSING"
                    job_record.finished_at = datetime.utcnow()
                    job_record.updated_at = datetime.utcnow()
                    session.add(job_record)
                    session.commit()
                return
            document_id = job.document_id or file_obj.document_id
            document = (
                session.get(DocumentRecord, document_id)
                if document_id is not None
                else None
            )
            if file_obj.status == FileStatus.COMPLETED:
                if job_record and job_record.status == JobStatus.QUEUED:
                    job_record.status = JobStatus.COMPLETED
                    job_record.finished_at = datetime.utcnow()
                    job_record.updated_at = datetime.utcnow()
                    session.add(job_record)
                    session.commit()
                return
            file_obj.status = FileStatus.PROCESSING
            file_obj.error = None
            file_obj.chunks = None
            file_obj.updated_at = datetime.utcnow()
            session.add(file_obj)
            if document:
                document.status = DocumentStatus.PROCESSING
                document.error = None
                document.chunks = None
                document.updated_at = datetime.utcnow()
                session.add(document)
            if job_record:
                job_record.status = JobStatus.PROCESSING
                job_record.started_at = datetime.utcnow()
                job_record.updated_at = datetime.utcnow()
                session.add(job_record)
            session.commit()

            page_ids = session.exec(
                select(PageRecord.id).where(PageRecord.file_id == file_obj.id)
            ).all()
            if page_ids:
                session.exec(delete(ChunkRecord).where(ChunkRecord.page_id.in_(page_ids)))
                session.exec(delete(PageRecord).where(PageRecord.id.in_(page_ids)))
                session.commit()

        success = True
        error_message: Optional[str] = None
        chunk_count = 0
        try:
            chunk_count = await self._ingest_file(job)
        except Exception as exc:
            success = False
            error_message = str(exc)
        finally:
            with Session(self.service.engine) as session:
                file_obj = session.get(FileRecord, job.file_id)
                document_id = job.document_id or file_obj.document_id
                document = (
                    session.get(DocumentRecord, document_id)
                    if document_id is not None
                    else None
                )
                job_record = (
                    session.get(JobRecord, job.job_record_id)
                    if job.job_record_id is not None
                    else None
                )
                if not file_obj:
                    return
                if success:
                    file_obj.status = FileStatus.COMPLETED
                    file_obj.updated_at = datetime.utcnow()
                    file_obj.error = None
                    file_obj.chunks = chunk_count
                    if document:
                        document.status = DocumentStatus.COMPLETED
                        document.updated_at = datetime.utcnow()
                        document.error = None
                        document.chunks = chunk_count
                        session.add(document)
                    if job_record:
                        job_record.status = JobStatus.COMPLETED
                        job_record.finished_at = datetime.utcnow()
                        job_record.updated_at = datetime.utcnow()
                        payload = dict(job_record.payload or {})
                        payload.update({"chunks": chunk_count, "attempt": job.attempt})
                        job_record.payload = payload
                        job_record.error = None
                        session.add(job_record)
                else:
                    file_obj.status = FileStatus.FAILED
                    file_obj.retries = job.attempt + 1
                    file_obj.updated_at = datetime.utcnow()
                    file_obj.error = error_message
                    file_obj.chunks = chunk_count or 0
                    if document:
                        document.status = DocumentStatus.FAILED
                        document.updated_at = datetime.utcnow()
                        document.error = error_message
                        document.chunks = chunk_count or 0
                        session.add(document)
                    if job_record:
                        job_record.status = JobStatus.FAILED
                        job_record.finished_at = datetime.utcnow()
                        job_record.updated_at = datetime.utcnow()
                        payload = dict(job_record.payload or {})
                        payload.update({"chunks": chunk_count or 0, "attempt": job.attempt})
                        job_record.payload = payload
                        job_record.error = error_message
                        session.add(job_record)
                session.add(file_obj)
                session.commit()

        if not success:
            await self._handle_failure(job)

    async def _ingest_file(self, job: IngestJob) -> int:
        with Session(self.service.engine) as session:
            file_obj = session.get(FileRecord, job.file_id)
            if not file_obj:
                return 0
            filename = file_obj.filename

        with open(job.path, "rb") as handle:
            pages = list(iter_document_pages(job.path, handle))

        chunk_payloads: list[dict[str, object]] = []
        with Session(self.service.engine) as session:
            batch_index = 0
            chunk_counter = 0
            for page_number, text in pages:
                page_sha = hashlib.sha256(
                    f"{job.sha256}:{page_number}:{len(text)}".encode("utf-8")
                ).hexdigest()
                try:
                    page_tokens = len(self._tokenizer.encode(text))
                except Exception:
                    page_tokens = len(text)
                page = PageRecord(
                    tenant_id=job.tenant_id,
                    file_id=job.file_id,
                    number=page_number,
                    sha256=page_sha,
                    text=text,
                    tokens=page_tokens,
                    meta={
                        "document_sha": job.sha256,
                        "page": page_number,
                        "file_id": job.file_id,
                    },
                )
                session.add(page)
                session.commit()
                session.refresh(page)

                chunks = _chunk(
                    text,
                    chunk=self.chunk_size,
                    overlap=self.overlap,
                    encoder=self._tokenizer,
                )
                for offset, chunk_text in enumerate(chunks, start=1):
                    chunk_sha = hashlib.sha256(
                        f"{job.sha256}:{page.number}:{offset}:{chunk_text}".encode("utf-8")
                    ).hexdigest()
                    try:
                        chunk_tokens = len(self._tokenizer.encode(chunk_text))
                    except Exception:
                        chunk_tokens = len(chunk_text)
                    chunk = ChunkRecord(
                        tenant_id=job.tenant_id,
                        page_id=page.id,
                        index=offset,
                        sha256=chunk_sha,
                        text=chunk_text,
                        batch=batch_index,
                        tokens=chunk_tokens,
                        meta={
                            "document_sha": job.sha256,
                            "page": page.number,
                            "chunk": offset,
                        },
                    )
                    session.add(chunk)
                    chunk_payloads.append(
                        {
                            "file": filename,
                            "page": page.number,
                            "sha256": chunk_sha,
                            "text": chunk_text,
                            "tokens": chunk_tokens,
                            "meta": chunk.meta or {},
                        }
                    )
                    chunk_counter += 1
                    if chunk_counter % self.embed_batch_size == 0:
                        batch_index += 1
                        session.commit()
                session.commit()

        vectorstore.index_chunks(chunk_payloads)
        return len(chunk_payloads)

    async def _handle_failure(self, job: IngestJob) -> None:
        job.attempt += 1
        if job.attempt > self.service.max_retries:
            return
        file_obj: Optional[FileRecord] = None
        with Session(self.service.engine) as session:
            file_obj = session.get(FileRecord, job.file_id)
            if file_obj:
                file_obj.status = FileStatus.QUEUED
                file_obj.retries = job.attempt
                file_obj.updated_at = datetime.utcnow()
                session.add(file_obj)
                document = (
                    session.get(DocumentRecord, file_obj.document_id)
                    if file_obj.document_id is not None
                    else None
                )
                if document:
                    document.status = DocumentStatus.QUEUED
                    document.error = None
                    document.chunks = None
                    document.updated_at = datetime.utcnow()
                    session.add(document)
                session.commit()
                session.refresh(file_obj)
        delay = self.service.backoff_seconds * (2 ** (job.attempt - 1))
        if delay:
            await asyncio.sleep(delay)
        if file_obj:
            await self.service.enqueue_job(file_obj, attempt=job.attempt)

    async def shutdown(self) -> None:
        """Process remaining jobs and stop the worker."""

        self._stop = True
        await self.drain()

    def stop(self) -> None:
        self._stop = True


__all__ = ["IngestJob", "IngestQueueFullError", "IngestService", "IngestWorker"]
