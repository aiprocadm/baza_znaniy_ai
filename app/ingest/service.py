"""Async ingestion queue and worker implementation."""

from __future__ import annotations

import asyncio
import hashlib
import math
import logging
import os
import re
from asyncio import QueueEmpty, QueueFull
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Optional, Tuple, TYPE_CHECKING

from sqlmodel import Session, delete, select
from sqlalchemy import text

from app._module_reset import ensure_core_modules

ensure_core_modules()

from app.core.config import get_settings
from app.core.datetime_utils import utc_now
from app.ingest.chunking import _chunk, _get_tokenizer, parse_document
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

try:  # pragma: no cover - optional dependency resolution
    from apscheduler.jobstores.base import JobLookupError as _ApsJobLookupError
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback

    class _ApsJobLookupError(Exception):
        """Fallback exception when APScheduler is unavailable."""


JobLookupError = _ApsJobLookupError

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
else:  # pragma: no cover - runtime placeholders
    AsyncIOScheduler = Any  # type: ignore
    CronTrigger = Any  # type: ignore
    IntervalTrigger = Any  # type: ignore


def _load_scheduler_artifacts():
    """Dynamically import APScheduler components when available."""

    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler as scheduler_cls
        from apscheduler.triggers.cron import CronTrigger as cron_cls
        from apscheduler.triggers.interval import IntervalTrigger as interval_cls
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "APScheduler is required for background ingest processing. "
            "Install it with 'pip install apscheduler'."
        ) from exc
    return scheduler_cls, cron_cls, interval_cls


def _extract_npa_fields(content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    text = content or ""

    def _pick(patterns: list[str], fallback_key: str | None = None) -> str | None:
        if fallback_key and metadata.get(fallback_key):
            return str(metadata[fallback_key]).strip()
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip()
        return None

    act_type = _pick(
        [
            r"(?:тип\s*акта|вид\s*нпа)\s*[:\-]\s*([^\n]+)",
            r"\b(федеральный закон|постановление|приказ|указ)\b",
        ],
        "act_type",
    )
    issuer = _pick([r"(?:орган\s*принятия|издатель|issuer)\s*[:\-]\s*([^\n]+)"], "issuer")
    reg_number = _pick(
        [r"(?:№|номер|reg(?:istration)?\s*number)\s*[:\-]?\s*([A-Za-zА-Яа-я0-9\-\/]+)"],
        "reg_number",
    )
    adoption_date = _pick(
        [r"(?:дата\s*принятия|adoption\s*date)\s*[:\-]\s*([0-9]{2}\.[0-9]{2}\.[0-9]{4})"],
        "adoption_date",
    )
    effective_date = _pick(
        [
            r"(?:дата\s*вступления\s*в\s*силу|effective\s*date)\s*[:\-]\s*([0-9]{2}\.[0-9]{2}\.[0-9]{4})"
        ],
        "effective_date",
    )
    revision = _pick([r"(?:редакция|revision)\s*[:\-]\s*([^\n]+)"], "revision")
    is_active_raw = metadata.get("is_active")
    is_active = (
        True
        if is_active_raw is None
        else str(is_active_raw).strip().lower() in {"1", "true", "yes", "да"}
    )

    return {
        "act_type": act_type,
        "issuer": issuer,
        "reg_number": reg_number,
        "adoption_date": adoption_date,
        "effective_date": effective_date,
        "revision": revision,
        "is_active": is_active,
    }


def _parse_act_date(raw: object) -> datetime | None:
    """Parse a Russian-format act date (``dd.mm.yyyy``) into a ``datetime``.

    Anything that is not a non-empty ``dd.mm.yyyy`` string — ``None``, blanks,
    non-strings and malformed dates — degrades to ``None`` rather than raising,
    matching the lenient metadata contract of :func:`_extract_npa_fields`
    (which is the sole producer of the strings fed here).
    """

    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.strptime(raw, "%d.%m.%Y")
    except ValueError:
        return None


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

    @staticmethod
    def _coerce_queue_size(raw: object, *, default: int, source: str) -> int:
        """Normalize queue size inputs and enforce sane defaults."""

        sentinel_values = {
            "unbounded",
            "unlimited",
            "infinite",
            "inf",
            "none",
            "no-limit",
            "nolimit",
        }
        if raw is None:
            return default
        if isinstance(raw, (int, float)):
            if math.isinf(raw):
                logger.info(
                    "Configured ingest queue size %s from %s is infinite; treating as unlimited",
                    raw,
                    source,
                )
                return 0
            if isinstance(raw, float) and math.isnan(raw):
                logger.warning(
                    "Configured ingest queue size %s from %s is NaN; using fallback %s",
                    raw,
                    source,
                    default,
                )
                return default
        if isinstance(raw, Decimal):
            if raw.is_infinite():
                logger.info(
                    "Configured ingest queue size %s from %s is infinite; treating as unlimited",
                    raw,
                    source,
                )
                return 0
            if not raw.is_finite():
                logger.warning(
                    "Configured ingest queue size %s from %s is not finite; using fallback %s",
                    raw,
                    source,
                    default,
                )
                return default
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in sentinel_values:
                return 0
            if not normalized:
                logger.warning(
                    "Empty ingest queue size from %s; using fallback %s",
                    source,
                    default,
                )
                return default
            raw = normalized
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError):
            logger.warning(
                "Invalid ingest queue size %r from %s; falling back to %s",
                raw,
                source,
                default,
            )
            return default
        if value < 0:
            logger.info(
                "Configured ingest queue size %s from %s is negative; treating as unlimited",
                value,
                source,
            )
            return 0
        return value

    @staticmethod
    def _resolve_retries(max_retries: Optional[int], settings: Any) -> int:
        """Resolve retry count with param > env(INGEST_MAX_RETRIES) > settings."""

        if max_retries is not None:
            return max_retries
        env_retries = os.getenv("INGEST_MAX_RETRIES")
        if env_retries is not None:
            try:
                return int(env_retries)
            except ValueError:
                logger.warning(
                    "Invalid ingest retry count %r from environment; using settings value",
                    env_retries,
                )
        return settings.ingest_max_retries

    @staticmethod
    def _resolve_backoff(backoff_seconds: Optional[float], settings: Any) -> float:
        """Resolve backoff with param > env(INGEST_BACKOFF_SECONDS/_BASE) > settings."""

        if backoff_seconds is not None:
            return backoff_seconds
        env_backoff = os.getenv("INGEST_BACKOFF_SECONDS") or os.getenv("INGEST_BACKOFF_BASE")
        if env_backoff is not None:
            try:
                return float(env_backoff)
            except ValueError:
                logger.warning(
                    "Invalid ingest backoff %r from environment; using settings value",
                    env_backoff,
                )
        return settings.ingest_backoff_seconds

    def _resolve_queue(
        self,
        queue: Optional[asyncio.Queue[Optional[IngestJob]]],
        queue_maxsize: Optional[int],
        settings: Any,
    ) -> tuple[asyncio.Queue[Optional[IngestJob]], int]:
        """Resolve the queue object and its bounded size across all sources.

        Precedence for the size: ``queue_maxsize`` param > a provided queue's
        own maxsize > env(INGEST_QUEUE_SIZE/INGEST_MAX_QUEUE) > settings.
        """

        env_queue_raw = None
        env_queue_source = "settings"
        for candidate_name in ("INGEST_QUEUE_SIZE", "INGEST_MAX_QUEUE"):
            raw_value = os.getenv(candidate_name)
            if raw_value is not None:
                env_queue_raw = raw_value
                env_queue_source = f"environment {candidate_name}"
                break

        default_queue_size = self._coerce_queue_size(
            env_queue_raw if env_queue_raw is not None else settings.ingest_queue_size,
            default=int(settings.ingest_queue_size),
            source=env_queue_source if env_queue_raw is not None else "settings",
        )

        queue_size_source = env_queue_source if env_queue_raw is not None else "settings"
        queue_size_raw: object = default_queue_size
        if queue_maxsize is not None:
            queue_size_source = "queue_maxsize parameter"
            queue_size_raw = queue_maxsize
        elif queue is not None:
            queue_size_source = "provided queue"
            queue_size_raw = getattr(queue, "maxsize", default_queue_size)

        queue_size = self._coerce_queue_size(
            queue_size_raw, default=default_queue_size, source=queue_size_source
        )

        if queue is not None:
            actual_maxsize = getattr(queue, "maxsize", None)
            if queue_maxsize is not None and actual_maxsize not in (None, queue_size):
                raise ValueError(
                    f"Provided queue has maxsize {actual_maxsize} but queue_maxsize "
                    f"{queue_size} was requested"
                )
            if actual_maxsize is not None:
                queue_size = self._coerce_queue_size(
                    actual_maxsize,
                    default=queue_size,
                    source="provided queue",
                )
            return queue, queue_size
        return asyncio.Queue(maxsize=queue_size), queue_size

    def __init__(
        self,
        *,
        queue: Optional[asyncio.Queue[Optional[IngestJob]]] = None,
        queue_maxsize: Optional[int] = None,
        max_retries: Optional[int] = None,
        backoff_seconds: Optional[float] = None,
        engine=None,
        auto_process: bool = False,
        use_local_queue: Optional[bool] = None,
    ) -> None:
        settings = get_settings()

        retries = self._resolve_retries(max_retries, settings)
        backoff = self._resolve_backoff(backoff_seconds, settings)

        self.queue: asyncio.Queue[Optional[IngestJob]]
        self.queue, queue_size = self._resolve_queue(queue, queue_maxsize, settings)

        self.queue_maxsize = queue_size
        self.max_retries = max(0, int(retries))
        self.backoff_seconds = max(0.0, float(backoff))
        self._engine = engine
        self.auto_process = auto_process
        if use_local_queue is None:
            use_local_queue = bool(getattr(settings, "ingest_use_local_queue", True))
        self.use_local_queue = bool(use_local_queue)
        self.worker: "IngestWorker" | None = None
        self._scheduler: AsyncIOScheduler | None = None
        self._worker_job_id: str | None = None
        self._maintenance_job_id: str | None = None
        self.worker_interval_seconds = max(0.1, float(settings.ingest_worker_interval_seconds))
        self.maintenance_cron = settings.ingest_maintenance_cron
        self.job_retention_days = max(1, int(settings.ingest_job_retention_days))
        self.processing_timeout_seconds = max(
            1.0, float(settings.ingest_processing_timeout_seconds)
        )

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
        _, cron_cls, interval_cls = _load_scheduler_artifacts()
        if self._worker_job_id is None:
            trigger = interval_cls(seconds=self.worker_interval_seconds)
            job_id = self._scheduler_job_name("worker")
            job = self._scheduler.add_job(
                self.worker.drain,
                trigger=trigger,
                id=job_id,
                max_instances=1,
                coalesce=True,
            )
            job_identifier = getattr(job, "id", None) if job is not None else None
            self._worker_job_id = job_identifier or job_id
        if self.maintenance_cron and self._maintenance_job_id is None:
            try:
                if hasattr(cron_cls, "from_crontab"):
                    trigger = cron_cls.from_crontab(self.maintenance_cron)
                else:
                    trigger = cron_cls(self.maintenance_cron)
            except ValueError:
                logger.error(
                    "Invalid ingest maintenance cron expression: %s",
                    self.maintenance_cron,
                )
            else:
                job_id = self._scheduler_job_name("maintenance")
                job = self._scheduler.add_job(
                    self.run_maintenance,
                    trigger=trigger,
                    id=job_id,
                    max_instances=1,
                    coalesce=True,
                )
                job_identifier = getattr(job, "id", None) if job is not None else None
                self._maintenance_job_id = job_identifier or job_id

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
        """Prune stale history and fail jobs that exceeded recovery timeout."""

        now = utc_now()
        cutoff = now - timedelta(days=self.job_retention_days)
        stale_processing_cutoff = now - timedelta(seconds=self.processing_timeout_seconds)
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
                    FileRecord.updated_at < stale_processing_cutoff,
                )
            ).all()

            for file_obj in stale_files:
                file_obj.status = FileStatus.FAILED
                file_obj.error = "STALE_PROCESSING"
                file_obj.updated_at = utc_now()
                session.add(file_obj)
                if file_obj.document_id is not None:
                    document = session.get(DocumentRecord, file_obj.document_id)
                    if document:
                        document.status = DocumentStatus.FAILED
                        document.error = "STALE_PROCESSING"
                        document.updated_at = utc_now()
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

    @staticmethod
    def _extract_row_id_and_status(row: object) -> tuple[Optional[int], str | None]:
        """Pull ``(id, status)`` from a raw result row across driver shapes.

        SQLModel/SQLAlchemy hand back different row types depending on the
        backend and version: a ``_mapping``-bearing row, a plain indexable
        tuple, or an attribute object. Normalise all three here.
        """

        mapping = getattr(row, "_mapping", None)
        if mapping is not None and "id" in mapping:
            return mapping["id"], mapping.get("status")
        if hasattr(row, "__getitem__"):
            try:
                job_id: Optional[int] = int(row[0])  # type: ignore[index]
            except Exception:
                job_id = None
            try:
                previous_status: str | None = str(row[1])  # type: ignore[index]
            except Exception:
                previous_status = None
            return job_id, previous_status
        return getattr(row, "id", None), getattr(row, "status", None)

    @staticmethod
    def _ingest_job_from_record(job_record: JobRecord) -> "IngestJob | None":
        """Build an :class:`IngestJob` from a claimed job record's payload.

        Returns ``None`` when the payload lacks a usable ``file_id`` so the
        caller can skip the job without claiming it.
        """

        payload = dict(job_record.payload or {})
        file_id = payload.get("file_id")
        if file_id is None:
            return None
        try:
            file_id_int = int(file_id)
        except (TypeError, ValueError):
            return None

        filename = str(payload.get("filename") or payload.get("file") or "")
        path = str(payload.get("path") or "")
        sha256 = str(payload.get("sha256") or "")
        document_id = payload.get("document_id")
        try:
            document_id_int = int(document_id) if document_id is not None else None
        except (TypeError, ValueError):
            document_id_int = None

        attempt_value = payload.get("attempt")
        if attempt_value is None:
            attempt_value = job_record.attempt
        try:
            attempt = int(attempt_value or 0)
        except (TypeError, ValueError):
            attempt = 0

        return IngestJob(
            tenant_id=job_record.tenant_id,
            path=path,
            sha256=sha256,
            file_id=file_id_int,
            filename=filename,
            document_id=document_id_int,
            job_record_id=job_record.id,
            attempt=attempt,
        )

    def dequeue_next_job(self, session: Session | None = None) -> IngestJob | None:
        """Atomically reserve the next queued job from the database."""

        manage_session = session is None
        session_obj = session or Session(self.engine)
        should_commit = False
        try:
            now = utc_now()
            stale_before = now - timedelta(seconds=self.processing_timeout_seconds)
            select_statement = text(
                """
                SELECT id, status
                FROM jobs
                WHERE job_type = :job_type
                  AND (
                    status = :queued
                    OR (
                        status = :processing
                        AND COALESCE(updated_at, started_at, created_at) <= :stale_before
                    )
                  )
                ORDER BY
                    CASE WHEN status = :queued THEN 0 ELSE 1 END,
                    priority DESC,
                    created_at
                LIMIT 1
                """
            )
            row = session_obj.exec(
                select_statement,
                params={
                    "job_type": "ingest",
                    "queued": JobStatus.QUEUED,
                    "processing": JobStatus.PROCESSING,
                    "stale_before": stale_before,
                },
            ).first()
            if row is None:
                return None

            job_id, previous_status = self._extract_row_id_and_status(row)
            if job_id is None or previous_status is None:
                return None

            # Optimistic-locking UPDATE: only claim the job if its status is
            # still what we observed in the SELECT above. If a competing worker
            # transitioned it first, rowcount == 0 and we yield this round.
            claim_statement = text(
                """
                UPDATE jobs
                SET status = :processing,
                    started_at = COALESCE(started_at, :ts),
                    updated_at = :ts
                WHERE id = :id AND status = :previous_status
                """
            )
            claim_result = session_obj.execute(
                claim_statement,
                {
                    "processing": JobStatus.PROCESSING,
                    "ts": now,
                    "id": job_id,
                    "previous_status": previous_status,
                },
            )
            if getattr(claim_result, "rowcount", 0) == 0:
                return None

            job_record = session_obj.get(JobRecord, job_id)
            if job_record is None:
                return None
            session_obj.refresh(job_record)
            recovered_stuck_job = previous_status == JobStatus.PROCESSING
            if recovered_stuck_job:
                job_record.attempt = int(job_record.attempt or 0) + 1
                job_record.error = "RECOVERED_STUCK_JOB"
                payload = dict(job_record.payload or {})
                payload.update(
                    {
                        "attempt": job_record.attempt,
                        "recovered_stuck_job": True,
                        "recovered_at": now.isoformat(),
                        "recovery_reason": "RECOVERED_STUCK_JOB",
                    }
                )
                job_record.payload = payload
                session_obj.add(job_record)

            job = self._ingest_job_from_record(job_record)
            if job is None:
                return None

            should_commit = True
            return job
        except Exception:
            if manage_session:
                session_obj.rollback()
            raise
        finally:
            if manage_session:
                try:
                    if should_commit:
                        session_obj.commit()
                    else:
                        session_obj.rollback()
                finally:
                    session_obj.close()

    async def enqueue_job(
        self, file_obj: FileRecord, *, attempt: int = 0, session: Session | None = None
    ) -> IngestJob:
        if self.use_local_queue and self.queue.full():
            raise IngestQueueFullError("INGEST_QUEUE_FULL")

        job = self._make_job(file_obj, attempt=attempt)
        manage_session = session is None
        session_obj = session or Session(self.engine)

        try:
            job_record = self._create_job_record(job, session=session_obj)
            job.job_record_id = job_record.id
            worker = getattr(self, "worker", None)
            if self.auto_process and worker is not None:
                commit = getattr(session_obj, "commit", None)
                if callable(commit):
                    commit()
                else:  # pragma: no cover - extremely defensive
                    session_obj.flush()
                await worker.process_job(job)
                return job
            if self.use_local_queue:
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
                    document.updated_at = utc_now()
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
                    file_obj.updated_at = utc_now()
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

    @staticmethod
    def _load_file_and_job(
        session: Session, job: IngestJob
    ) -> "tuple[FileRecord | None, JobRecord | None]":
        """Load the file row and (optional) job record for a job in one place.

        ``job_record`` is only fetched when the job carries a record id, mirroring
        the original inline loads in :meth:`_process`.
        """

        file_obj = session.get(FileRecord, job.file_id)
        job_record = (
            session.get(JobRecord, job.job_record_id) if job.job_record_id is not None else None
        )
        return file_obj, job_record

    @staticmethod
    def _fail_job_file_missing(session: Session, job_record: "JobRecord | None") -> None:
        """Mark a claimed job FAILED/FILE_MISSING when its file row has vanished.

        No-op when there is no job record to update. Commits its own change so
        callers can simply ``return`` afterwards.
        """

        if job_record is None:
            return
        now = utc_now()
        job_record.status = JobStatus.FAILED
        job_record.error = "FILE_MISSING"
        job_record.finished_at = now
        job_record.updated_at = now
        session.add(job_record)
        session.commit()

    @staticmethod
    def _finalize_job(
        session: Session,
        *,
        file_obj: FileRecord,
        document: "DocumentRecord | None",
        job_record: "JobRecord | None",
        success: bool,
        chunk_count: int,
        error_message: Optional[str],
        attempt: int,
    ) -> None:
        """Write terminal status onto the file/document/job rows after ingest.

        Consolidates the symmetric success/failure branches of :meth:`_process`'s
        finalization. On failure ``file_obj.retries`` advances to ``attempt + 1``
        and the error message is propagated to every row; on success the error is
        cleared. Commits once at the end.
        """

        now = utc_now()
        file_status = FileStatus.COMPLETED if success else FileStatus.FAILED
        doc_status = DocumentStatus.COMPLETED if success else DocumentStatus.FAILED
        job_status = JobStatus.COMPLETED if success else JobStatus.FAILED
        chunks_value = chunk_count if success else (chunk_count or 0)
        error_value = None if success else error_message

        file_obj.status = file_status
        file_obj.updated_at = now
        file_obj.error = error_value
        file_obj.chunks = chunks_value
        if not success:
            file_obj.retries = attempt + 1
        session.add(file_obj)

        if document:
            document.status = doc_status
            document.updated_at = now
            document.error = error_value
            document.chunks = chunks_value
            session.add(document)

        if job_record:
            job_record.status = job_status
            job_record.finished_at = now
            job_record.updated_at = now
            payload = dict(job_record.payload or {})
            payload.update({"chunks": chunks_value, "attempt": attempt})
            job_record.payload = payload
            job_record.error = error_value
            session.add(job_record)

        session.commit()

    async def process_job(self, job: IngestJob) -> None:
        logger.debug("ingest worker immediate processing job %s", job)
        try:
            await self._process(job)
        except Exception:
            await self._handle_failure(job)

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
            job: IngestJob | None = None
            from_queue = False
            if self.service.use_local_queue:
                try:
                    job = self.service.queue.get_nowait()
                    from_queue = True
                except QueueEmpty:
                    job = None
            if job is None and from_queue:
                self.service.queue.task_done()
                continue
            if job is None:
                job = await asyncio.to_thread(self.service.dequeue_next_job)
                if job is None:
                    break
            logger.debug("ingest worker processing job %s", job)
            try:
                await self._process(job)
            except Exception:  # pragma: no cover - defensive
                await self._handle_failure(job)
            finally:
                if from_queue:
                    self.service.queue.task_done()
            processed += 1
        return processed

    def _begin_processing(self, job: IngestJob) -> bool:
        """Reserve a job: flip its rows to PROCESSING and clear stale pages.

        Returns ``False`` (caller should stop) when the file row has vanished or
        the file is already COMPLETED; ``True`` when ingestion should proceed.
        """

        with Session(self.service.engine) as session:
            file_obj, job_record = self._load_file_and_job(session, job)
            if not file_obj:
                self._fail_job_file_missing(session, job_record)
                return False
            document_id = job.document_id or file_obj.document_id
            document = session.get(DocumentRecord, document_id) if document_id is not None else None
            if file_obj.status == FileStatus.COMPLETED:
                if job_record and job_record.status == JobStatus.QUEUED:
                    job_record.status = JobStatus.COMPLETED
                    job_record.finished_at = utc_now()
                    job_record.updated_at = utc_now()
                    session.add(job_record)
                    session.commit()
                return False
            file_obj.status = FileStatus.PROCESSING
            file_obj.error = None
            file_obj.chunks = None
            file_obj.updated_at = utc_now()
            session.add(file_obj)
            if document:
                document.status = DocumentStatus.PROCESSING
                document.error = None
                document.chunks = None
                document.updated_at = utc_now()
                session.add(document)
            if job_record:
                job_record.status = JobStatus.PROCESSING
                job_record.started_at = utc_now()
                job_record.updated_at = utc_now()
                session.add(job_record)
            session.commit()

            # scalars()/execute() are the SQLAlchemy-native equivalents of
            # SQLModel's exec() wrapper: scalars() yields the bare ids for the
            # single-column select, execute() runs the bulk-delete DML.
            page_ids = session.scalars(
                select(PageRecord.id).where(PageRecord.file_id == file_obj.id)
            ).all()
            if page_ids:
                session.execute(delete(ChunkRecord).where(ChunkRecord.page_id.in_(page_ids)))
                session.execute(delete(PageRecord).where(PageRecord.id.in_(page_ids)))
                session.commit()
        return True

    def _finalize(
        self,
        job: IngestJob,
        *,
        success: bool,
        chunk_count: int,
        error_message: Optional[str],
    ) -> None:
        """Reload the rows and write terminal status after an ingest attempt."""

        with Session(self.service.engine) as session:
            file_obj, job_record = self._load_file_and_job(session, job)
            if not file_obj:
                self._fail_job_file_missing(session, job_record)
                return
            document_id = job.document_id or file_obj.document_id
            document = (
                session.get(DocumentRecord, document_id) if document_id is not None else None
            )
            self._finalize_job(
                session,
                file_obj=file_obj,
                document=document,
                job_record=job_record,
                success=success,
                chunk_count=chunk_count,
                error_message=error_message,
                attempt=job.attempt,
            )

    async def _process(self, job: IngestJob) -> None:
        if not self._begin_processing(job):
            return

        success = True
        error_message: Optional[str] = None
        chunk_count = 0
        try:
            chunk_count = await self._ingest_file(job)
        except Exception as exc:
            success = False
            error_message = str(exc)
            logger.exception("Failed to ingest job %s", job)

        # _finalize runs only when control falls through normally. A
        # BaseException (e.g. asyncio.CancelledError on shutdown) is not caught
        # by ``except Exception``, so it propagates here, skipping finalization
        # and leaving the job PROCESSING for stale-recovery to reclaim. Keeping
        # this out of a ``finally`` is what stops a ``return`` swallowing it.
        self._finalize(
            job,
            success=success,
            chunk_count=chunk_count,
            error_message=error_message,
        )

        if not success:
            await self._handle_failure(job)

    @staticmethod
    def _count_tokens(encoder: Any, text: str) -> int:
        """Token count for ``text``, falling back to character length.

        The tokenizer can be a stub or raise on odd input; degrade to
        ``len(text)`` rather than failing the whole ingest.
        """

        try:
            return len(encoder.encode(text))
        except Exception:
            return len(text)

    @staticmethod
    def _page_sha(document_sha: str, page_number: int, text: str) -> str:
        """Stable page digest keyed on the document sha, page number and length."""

        return hashlib.sha256(
            f"{document_sha}:{page_number}:{len(text)}".encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _chunk_sha(document_sha: str, page_number: int, offset: int, chunk_text: str) -> str:
        """Stable chunk digest keyed on the document sha, page, offset and text."""

        return hashlib.sha256(
            f"{document_sha}:{page_number}:{offset}:{chunk_text}".encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _chunk_meta(
        document_sha: str,
        page_number: int,
        offset: int,
        parse_result: Any,
        attrs: dict[str, Any],
    ) -> dict[str, Any]:
        """Per-chunk metadata: parse provenance merged with the document NPA attrs."""

        return {
            "document_sha": document_sha,
            "page": page_number,
            "chunk": offset,
            "parser_backend": parse_result.parser_backend_used,
            "fallback_reason": parse_result.fallback_reason,
            "ocr_used": parse_result.ocr_used,
            **attrs,
        }

    @staticmethod
    def _build_chunk_payload(
        *,
        filename: str,
        page_number: int,
        chunk_sha: str,
        chunk_text: str,
        chunk_tokens: int,
        meta: dict[str, Any] | None,
        tenant_id: str,
    ) -> dict[str, object]:
        """Assemble the vectorstore index payload for a single chunk."""

        return {
            "file": filename,
            "page": page_number,
            "sha256": chunk_sha,
            "text": chunk_text,
            "tokens": chunk_tokens,
            "meta": meta or {},
            "owner": tenant_id,
            "tenant_id": tenant_id,
        }

    async def _ingest_file(self, job: IngestJob) -> int:
        with Session(self.service.engine) as session:
            file_obj = session.get(FileRecord, job.file_id)
            if not file_obj:
                return 0
            filename = file_obj.filename

        with open(job.path, "rb") as handle:
            parse_result = parse_document(job.path, handle)
            pages = list(parse_result.pages)

        chunk_payloads: list[dict[str, object]] = []
        full_text = "\n".join(text for _page, text in pages)
        with Session(self.service.engine) as session:
            document = (
                session.get(DocumentRecord, job.document_id)
                if job.document_id is not None
                else None
            )
            attrs: dict[str, Any] = {}
            if document is not None:
                attrs = _extract_npa_fields(full_text, metadata=document.meta or {})
                document.content = full_text
                document.act_type = attrs.get("act_type")
                document.issuer = attrs.get("issuer")
                document.reg_number = attrs.get("reg_number")
                document.revision = attrs.get("revision")
                document.is_active = bool(attrs.get("is_active", True))
                for field_name in ("adoption_date", "effective_date"):
                    setattr(document, field_name, _parse_act_date(attrs.get(field_name)))
                session.add(document)
                session.commit()
            batch_index = 0
            chunk_counter = 0
            for page_number, text in pages:
                page_sha = self._page_sha(job.sha256, page_number, text)
                page_tokens = self._count_tokens(self._tokenizer, text)
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
                        "parser_backend": parse_result.parser_backend_used,
                        "fallback_reason": parse_result.fallback_reason,
                        "ocr_used": parse_result.ocr_used,
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
                    chunk_sha = self._chunk_sha(job.sha256, page.number, offset, chunk_text)
                    chunk_tokens = self._count_tokens(self._tokenizer, chunk_text)
                    chunk_meta = self._chunk_meta(
                        job.sha256, page.number, offset, parse_result, attrs
                    )
                    chunk = ChunkRecord(
                        tenant_id=job.tenant_id,
                        page_id=page.id,
                        index=offset,
                        sha256=chunk_sha,
                        text=chunk_text,
                        batch=batch_index,
                        tokens=chunk_tokens,
                        meta=chunk_meta,
                    )
                    session.add(chunk)
                    chunk_payloads.append(
                        self._build_chunk_payload(
                            filename=filename,
                            page_number=page.number,
                            chunk_sha=chunk_sha,
                            chunk_text=chunk_text,
                            chunk_tokens=chunk_tokens,
                            meta=chunk_meta,
                            tenant_id=job.tenant_id,
                        )
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
                file_obj.updated_at = utc_now()
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
                    document.updated_at = utc_now()
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
