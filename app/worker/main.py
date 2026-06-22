"""Dedicated entrypoint for running the asynchronous ingest worker."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import suppress
from datetime import timezone
from functools import partial

from app.core.config import get_settings
from app.ingest import IngestService, IngestWorker
from app.observability.logging import bind_log_context, configure_structured_logging
from app.observability.metadata_guard import schedule_sqlmodel_metadata_guard

try:  # pragma: no cover - optional dependency resolution
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
except ModuleNotFoundError:  # pragma: no cover - scheduler is optional
    AsyncIOScheduler = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
_shutdown_event = asyncio.Event()


def _handle_signal(signum: int) -> None:
    """Trigger a graceful shutdown when receiving *signum*."""

    bind_log_context(task_id="worker-signal")
    logger.info("Received signal %s; shutting down ingest worker", signum)
    _shutdown_event.set()


def _install_signal_handlers() -> None:
    """Register signal handlers for CTRL+C and termination requests."""

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, partial(_handle_signal, sig))
        except NotImplementedError:  # pragma: no cover - Windows fallback
            signal.signal(sig, lambda *_args, _sig=sig: _handle_signal(_sig))
        except RuntimeError:  # pragma: no cover - loop not running yet
            signal.signal(sig, lambda *_args, _sig=sig: _handle_signal(_sig))


async def _manual_poll_loop(worker: IngestWorker, interval: float) -> None:
    """Fallback polling loop when APScheduler is not installed."""

    maintenance_task = asyncio.create_task(_maintenance_loop(worker.service))
    try:
        while not _shutdown_event.is_set():
            processed = await worker.drain()
            if processed == 0:
                await asyncio.sleep(interval)
    finally:
        maintenance_task.cancel()
        with suppress(asyncio.CancelledError):
            await maintenance_task
        await worker.shutdown()


async def _maintenance_loop(service: IngestService) -> None:
    """Execute the maintenance job at the configured cron interval."""

    settings = get_settings()
    interval = max(60.0, settings.ingest_worker_interval_seconds * 30)
    while not _shutdown_event.is_set():
        await asyncio.sleep(interval)
        try:
            await service.run_maintenance()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Ingest maintenance task failed")


async def _run_worker() -> None:
    """Configure the ingest worker and block until shutdown."""

    settings = get_settings()
    ingest_service = IngestService(
        max_retries=settings.ingest_max_retries,
        backoff_seconds=settings.ingest_backoff_seconds,
        auto_process=True,
        use_local_queue=settings.ingest_use_local_queue,
    )
    ingest_worker = IngestWorker(ingest_service)
    ingest_service.set_worker(ingest_worker)

    scheduler: AsyncIOScheduler | None = None
    if AsyncIOScheduler is not None:
        scheduler = AsyncIOScheduler(timezone=timezone.utc)
        ingest_service.configure_scheduler(scheduler)
        schedule_sqlmodel_metadata_guard(scheduler)
        try:
            ingest_service.ensure_background_worker()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to initialise background worker schedule")
        try:
            scheduler.start()
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to start ingest scheduler")
            scheduler = None
    else:
        logger.warning("APScheduler is not installed; falling back to manual ingest polling")

    if scheduler is None:
        await _manual_poll_loop(ingest_worker, max(0.5, settings.ingest_worker_interval_seconds))
    else:
        await ingest_worker.drain()
        await _shutdown_event.wait()
        with suppress(Exception):
            await scheduler.shutdown(wait=True)
        await ingest_worker.shutdown()


async def amain() -> None:
    """Async entrypoint for the standalone worker process."""

    configure_structured_logging(level=logging.INFO)
    bind_log_context(task_id="ingest-worker")
    _install_signal_handlers()
    try:
        await _run_worker()
    finally:
        _shutdown_event.set()


def main() -> None:
    """CLI entrypoint for the ingest worker container."""

    asyncio.run(amain())


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()
