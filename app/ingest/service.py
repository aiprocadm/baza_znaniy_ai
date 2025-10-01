"""Async ingestion queue and worker implementation."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from sqlmodel import Session, delete, select

from app.core.config import get_settings
from app.ingest.chunking import _chunk, _get_tokenizer, iter_document_pages
from app.models.entities import JobRecord, JobStatus
from app.models.file import (
    ChunkRecord,
    DocumentRecord,
    DocumentStatus,
    FileRecord,
    FileStatus,
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


class IngestService:
    """Service responsible for computing file hashes and enqueueing jobs."""

    def __init__(
        self,
        *,
        queue: Optional[asyncio.Queue[Optional[IngestJob]]] = None,
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
        self.queue: asyncio.Queue[Optional[IngestJob]] = queue or asyncio.Queue()
        self.max_retries = max(0, int(retries))
        self.backoff_seconds = max(0.0, float(backoff))
        self._engine = engine
        self.auto_process = auto_process
        self._thread_lock = threading.Lock()
        self._worker_instance: Optional["IngestWorker"] = None
        self._job_threads: set[threading.Thread] = set()

        self.worker: "IngestWorker" | None = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = get_engine()
        return self._engine

    def set_worker(self, worker: "IngestWorker") -> None:
        """Register the worker instance responsible for background processing."""

        with self._thread_lock:
            self._worker_instance = worker

    def ensure_background_worker(self) -> None:
        """Start a background worker thread when auto-processing is enabled."""

        if not self.auto_process:
            return
        with self._thread_lock:
            if self._worker_instance is None:
                self._worker_instance = IngestWorker(self)
            # Prune completed job threads to avoid unbounded growth.
            self._job_threads = {thread for thread in self._job_threads if thread.is_alive()}

    async def stop_background_worker(self) -> None:
        """Signal the background worker thread to shut down."""

        if not self.auto_process:
            return
        # Wait for active ingestion threads to finish gracefully.
        while True:
            with self._thread_lock:
                threads = [thread for thread in self._job_threads if thread.is_alive()]
                if not threads:
                    self._job_threads.clear()
                    break
            for thread in threads:
                thread.join(timeout=1)

    def _spawn_job_thread(self, job: IngestJob) -> None:
        """Execute a job asynchronously using a dedicated worker thread."""

        self.ensure_background_worker()
        worker = self._worker_instance or IngestWorker(self)

        def _runner() -> None:
            try:
                asyncio.run(worker._process(job))
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Background ingest job failed for file %s", job.file_id)
            finally:
                with self._thread_lock:
                    self._job_threads.discard(threading.current_thread())

        thread = threading.Thread(
            target=_runner,
            name=f"ingest-job-{job.file_id}",
            daemon=True,
        )
        with self._thread_lock:
            self._job_threads.add(thread)
        thread.start()
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

    def _create_job_record(self, job: IngestJob) -> JobRecord:
        payload = {
            "file_id": job.file_id,
            "sha256": job.sha256,
            "path": job.path,
            "filename": job.filename,
            "attempt": job.attempt,
        }
        if job.document_id is not None:
            payload["document_id"] = job.document_id

        with Session(self.engine) as session:
            record = JobRecord(
                tenant_id=job.tenant_id,
                job_type="ingest",
                status=JobStatus.QUEUED,
                priority=0,
                error=None,
                resource_id=str(job.file_id),
                attempt=job.attempt,
                payload=payload,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    async def enqueue_job(self, file_obj: FileRecord, *, attempt: int = 0) -> IngestJob:
        job = self._make_job(file_obj, attempt=attempt)
        job_record = self._create_job_record(job)
        job.job_record_id = job_record.id
        if self.auto_process:
            self._spawn_job_thread(job)
        else:
            await self.queue.put(job)

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
            document = session.exec(
                select(DocumentRecord).where(DocumentRecord.sha256 == sha)
            ).first()
            if document is None:
                document = DocumentRecord(
                    sha256=sha,
                    mime_type=detected_mime,
                    status=DocumentStatus.QUEUED,
                    error=None,
                    chunks=None,
                )
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
            session.commit()
            session.refresh(file_obj)
            session.refresh(document)

        if queued:
            await self.enqueue_job(file_obj)
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
        self._task: asyncio.Task[None] | None = None
        self.service.worker = self

    def ensure_started(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - synchronous fallback
            return
        self._task = loop.create_task(self.run())

    async def run(self) -> None:
        logger.debug("ingest worker run loop entered")
        while True:
            if self._stop and self.service.queue.empty():
                break
            try:
                job = await self.service.queue.get()
            except asyncio.CancelledError:  # pragma: no cover - shutdown safety
                break
            if job is None:
                self.service.queue.task_done()
                break
            logger.debug("ingest worker processing job %s", job)
            try:
                await self._process(job)
            except Exception:  # pragma: no cover - defensive
                await self._handle_failure(job)
            finally:
                self.service.queue.task_done()

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

    def stop(self) -> None:
        self._stop = True
        task = self._task
        if task is not None:
            task.cancel()


__all__ = ["IngestJob", "IngestService", "IngestWorker"]
