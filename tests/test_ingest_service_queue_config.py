import asyncio
from decimal import Decimal
from types import SimpleNamespace

from app.ingest import service as service_module


def test_coerce_queue_size_handles_various_inputs():
    assert service_module.IngestService._coerce_queue_size(None, default=5, source="default") == 5
    assert service_module.IngestService._coerce_queue_size("  unbounded ", default=5, source="env") == 0
    assert service_module.IngestService._coerce_queue_size(-3, default=5, source="cfg") == 0
    assert service_module.IngestService._coerce_queue_size("7", default=5, source="cfg") == 7
    assert service_module.IngestService._coerce_queue_size(Decimal("NaN"), default=5, source="cfg") == 5
    assert service_module.IngestService._coerce_queue_size("bad", default=5, source="cfg") == 5


def test_queue_size_from_environment(monkeypatch):
    from app.core import config as config_module

    monkeypatch.setenv("INGEST_QUEUE_SIZE", "unlimited")
    config_module.get_settings.cache_clear()

    async def scenario() -> None:
        service = service_module.IngestService(max_retries=0, backoff_seconds=0)
        assert service.queue_maxsize == 0
        assert service.queue.maxsize == 0

    try:
        asyncio.run(scenario())
    finally:
        config_module.get_settings.cache_clear()
        monkeypatch.delenv("INGEST_QUEUE_SIZE", raising=False)


class DummyScheduler:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, SimpleNamespace]] = []
        self.removed: list[str] = []

    def add_job(self, func, *, trigger, id, max_instances, coalesce):
        job = SimpleNamespace(func=func, trigger=trigger, id=id)
        self.jobs.append((id, job))
        return job

    def remove_job(self, job_id: str) -> None:
        self.removed.append(job_id)


class DummyIntervalTrigger:
    def __init__(self, *, seconds: float) -> None:
        self.seconds = seconds


class DummyCronTrigger:
    def __init__(self, expression: str) -> None:
        self.expression = expression

    @classmethod
    def from_crontab(cls, expression: str) -> "DummyCronTrigger":
        if "invalid" in expression:
            raise ValueError("invalid cron")
        return cls(expression)


class DummyWorker:
    def __init__(self) -> None:
        self.shutdown_called = False

    async def drain(self) -> None:  # pragma: no cover - invoked by scheduler in production
        return None

    async def shutdown(self) -> None:
        self.shutdown_called = True


def test_background_worker_lifecycle(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "_load_scheduler_artifacts",
        lambda: (None, DummyCronTrigger, DummyIntervalTrigger),
    )

    async def scenario() -> None:
        service = service_module.IngestService(auto_process=True, max_retries=0, backoff_seconds=0)
        service.worker = DummyWorker()
        service.maintenance_cron = "*/5 * * * *"

        scheduler = DummyScheduler()
        service.configure_scheduler(scheduler)

        service.ensure_background_worker()

        assert any(job_id.endswith("worker") for job_id, _ in scheduler.jobs)
        assert any(job_id.endswith("maintenance") for job_id, _ in scheduler.jobs)

        await service.stop_background_worker()

        assert service.worker.shutdown_called is True
        assert scheduler.removed

    asyncio.run(scenario())


def test_background_worker_skips_invalid_cron(monkeypatch):
    captured: list[str] = []

    def fake_logger_error(message, cron_expression, *args, **kwargs):
        captured.append(cron_expression)

    monkeypatch.setattr(service_module.logger, "error", fake_logger_error)
    monkeypatch.setattr(
        service_module,
        "_load_scheduler_artifacts",
        lambda: (None, DummyCronTrigger, DummyIntervalTrigger),
    )

    async def scenario() -> None:
        service = service_module.IngestService(auto_process=True, max_retries=0, backoff_seconds=0)
        service.worker = DummyWorker()
        service.maintenance_cron = "invalid cron"

        scheduler = DummyScheduler()
        service.configure_scheduler(scheduler)

        service.ensure_background_worker()

        assert any(job_id.endswith("worker") for job_id, _ in scheduler.jobs)
        assert not any(job_id.endswith("maintenance") for job_id, _ in scheduler.jobs)
        assert captured == ["invalid cron"]

        await service.stop_background_worker()

    asyncio.run(scenario())


