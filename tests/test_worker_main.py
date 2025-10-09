import asyncio
import importlib
import signal
import sys
from collections.abc import Callable
from types import ModuleType

import pytest


def _reload_worker_module():
    """Return a freshly reloaded ``app.worker.main`` module."""

    return importlib.reload(importlib.import_module("app.worker.main"))


def test_handle_signal_sets_shutdown_event(monkeypatch):
    worker = _reload_worker_module()

    async def scenario() -> None:
        event = asyncio.Event()
        monkeypatch.setattr(worker, "_shutdown_event", event)
        worker._handle_signal(signal.SIGTERM)
        assert event.is_set()

    asyncio.run(scenario())


def test_install_signal_handlers_uses_event_loop_handlers(monkeypatch):
    worker = _reload_worker_module()
    captured: list[int] = []

    def fake_handle(sig: int) -> None:
        captured.append(sig)

    monkeypatch.setattr(worker, "_handle_signal", fake_handle)

    class DummyLoop:
        def __init__(self) -> None:
            self.handlers: dict[int, Callable[[], None]] = {}

        def add_signal_handler(self, sig: signal.Signals, callback):  # type: ignore[override]
            self.handlers[int(sig)] = callback

    loop = DummyLoop()
    monkeypatch.setattr(worker.asyncio, "get_running_loop", lambda: loop)

    worker._install_signal_handlers()

    assert set(loop.handlers) == {signal.SIGTERM, signal.SIGINT}

    for sig, callback in loop.handlers.items():
        callback()
        assert captured[-1] == sig


def test_install_signal_handlers_fallback(monkeypatch):
    worker = _reload_worker_module()
    recorded: list[tuple[int, Callable[[], None]]] = []
    captured: list[int] = []

    def fake_signal(sig: signal.Signals, handler):  # type: ignore[override]
        recorded.append((int(sig), handler))

    def fake_handle(sig: int) -> None:
        captured.append(sig)

    class FailingLoop:
        def add_signal_handler(self, sig: signal.Signals, _callback):  # type: ignore[override]
            raise NotImplementedError

    monkeypatch.setattr(worker.asyncio, "get_running_loop", lambda: FailingLoop())
    monkeypatch.setattr(worker.signal, "signal", fake_signal)
    monkeypatch.setattr(worker, "_handle_signal", fake_handle)

    worker._install_signal_handlers()

    assert {sig for sig, _ in recorded} == {int(signal.SIGTERM), int(signal.SIGINT)}
    for sig, handler in recorded:
        handler()
        assert captured[-1] == sig


def test_manual_poll_loop_processes_until_shutdown(monkeypatch):
    worker = _reload_worker_module()

    async def scenario() -> None:
        event = asyncio.Event()
        monkeypatch.setattr(worker, "_shutdown_event", event)
        durations: list[float] = []
        loop_started = asyncio.Event()

        async def fake_sleep(delay: float) -> None:
            durations.append(delay)
            await loop_started.wait()
            event.set()

        async def fake_maintenance_loop(service) -> None:
            service.maintenance_started = True
            loop_started.set()
            await event.wait()

        monkeypatch.setattr(worker.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(worker, "_maintenance_loop", fake_maintenance_loop)

        class DummyService:
            def __init__(self) -> None:
                self.maintenance_started = False

        class DummyWorker:
            def __init__(self) -> None:
                self.service = DummyService()
                self.shutdown_called = False
                self.calls = 0

            async def drain(self) -> int:
                self.calls += 1
                return 0 if self.calls == 1 else 1

            async def shutdown(self) -> None:
                self.shutdown_called = True

        worker_instance = DummyWorker()

        await worker._manual_poll_loop(worker_instance, interval=0.75)

        assert durations == [0.75]
        assert worker_instance.shutdown_called
        assert worker_instance.service.maintenance_started

    asyncio.run(scenario())


def test_maintenance_loop_invokes_service_and_logs(monkeypatch):
    worker = _reload_worker_module()

    async def scenario() -> None:
        event = asyncio.Event()
        monkeypatch.setattr(worker, "_shutdown_event", event)
        sleep_calls: list[float] = []
        logged_messages: list[str] = []

        class DummySettings:
            ingest_worker_interval_seconds = 1.0

        class DummyService:
            async def run_maintenance(self) -> None:
                event.set()
                raise RuntimeError("boom")

        async def fake_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        def fake_exception(msg: str) -> None:
            logged_messages.append(msg)

        monkeypatch.setattr(worker, "get_settings", lambda: DummySettings())
        monkeypatch.setattr(worker.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(worker.logger, "exception", fake_exception)

        await worker._maintenance_loop(DummyService())

        assert sleep_calls == [max(60.0, DummySettings.ingest_worker_interval_seconds * 30)]
        assert logged_messages == ["Ingest maintenance task failed"]

    asyncio.run(scenario())


def test_run_worker_with_scheduler(monkeypatch):
    worker = _reload_worker_module()

    async def scenario() -> None:
        event = asyncio.Event()
        monkeypatch.setattr(worker, "_shutdown_event", event)
        created: dict[str, object] = {}
        guard_calls: list[object] = []

        class DummySettings:
            ingest_worker_interval_seconds = 1.5
            ingest_max_retries = 2
            ingest_backoff_seconds = 3.0
            ingest_use_local_queue = True

        class DummyScheduler:
            def __init__(self, timezone):
                self.timezone = timezone
                self.started = False
                self.shutdown_args: tuple[bool, ...] | None = None

            def start(self) -> None:
                self.started = True

            async def shutdown(self, wait: bool) -> None:
                self.shutdown_args = (wait,)

        class DummyService:
            def __init__(self, **kwargs):
                created["service_kwargs"] = kwargs
                created["service"] = self
                self.worker = None
                self.scheduler = None

            def set_worker(self, worker_instance) -> None:
                self.worker = worker_instance

            def configure_scheduler(self, scheduler) -> None:
                self.scheduler = scheduler

            def ensure_background_worker(self) -> None:
                created["ensured"] = True

        class DummyWorker:
            def __init__(self, service):
                created["worker"] = self
                self.service = service
                self.shutdown_called = False
                self.drain_calls = 0

            async def drain(self) -> None:
                self.drain_calls += 1
                asyncio.get_running_loop().call_soon(event.set)

            async def shutdown(self) -> None:
                self.shutdown_called = True

        monkeypatch.setattr(worker, "get_settings", lambda: DummySettings())
        monkeypatch.setattr(worker, "AsyncIOScheduler", DummyScheduler)
        monkeypatch.setattr(worker, "IngestService", DummyService)
        monkeypatch.setattr(worker, "IngestWorker", DummyWorker)
        monkeypatch.setattr(
            worker, "schedule_sqlmodel_metadata_guard", lambda scheduler: guard_calls.append(scheduler)
        )

        await worker._run_worker()

        service = created["service"]
        worker_instance = created["worker"]
        scheduler = service.scheduler

        assert created["service_kwargs"] == {
            "max_retries": DummySettings.ingest_max_retries,
            "backoff_seconds": DummySettings.ingest_backoff_seconds,
            "auto_process": True,
            "use_local_queue": DummySettings.ingest_use_local_queue,
        }
        assert service.worker is worker_instance
        assert created["ensured"] is True
        assert scheduler.started is True
        assert guard_calls == [scheduler]
        assert scheduler.shutdown_args == (True,)
        assert worker_instance.shutdown_called is True
        assert worker_instance.drain_calls == 1

    asyncio.run(scenario())


def test_run_worker_manual_poll_when_scheduler_missing(monkeypatch):
    worker = _reload_worker_module()

    async def scenario() -> None:
        event = asyncio.Event()
        monkeypatch.setattr(worker, "_shutdown_event", event)
        monkeypatch.setattr(worker, "AsyncIOScheduler", None)
        drain_calls: list[object] = []

        class DummySettings:
            ingest_worker_interval_seconds = 0.25
            ingest_max_retries = 1
            ingest_backoff_seconds = 0.1
            ingest_use_local_queue = False

        class DummyWorker:
            def __init__(self, service):
                self.service = service
                self.shutdown_called = False

            async def drain(self) -> int:
                drain_calls.append(True)
                event.set()
                return 0

            async def shutdown(self) -> None:
                self.shutdown_called = True

        class DummyService:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.worker = None

            def set_worker(self, worker_instance) -> None:
                self.worker = worker_instance

        async def fake_manual_poll(worker_instance, interval: float) -> None:
            assert interval == 0.5  # max(0.5, settings.ingest_worker_interval_seconds)
            await worker_instance.shutdown()

        monkeypatch.setattr(worker, "get_settings", lambda: DummySettings())
        monkeypatch.setattr(worker, "IngestService", DummyService)
        monkeypatch.setattr(worker, "IngestWorker", DummyWorker)
        monkeypatch.setattr(worker, "_manual_poll_loop", fake_manual_poll)

        await worker._run_worker()

        assert drain_calls == []

    asyncio.run(scenario())


def test_amain_invokes_worker_and_sets_event(monkeypatch):
    worker = _reload_worker_module()

    async def scenario() -> None:
        event = asyncio.Event()
        monkeypatch.setattr(worker, "_shutdown_event", event)
        calls: list[str] = []

        async def fake_run_worker() -> None:
            calls.append("run_worker")

        monkeypatch.setattr(worker, "_install_signal_handlers", lambda: calls.append("install"))
        monkeypatch.setattr(worker, "_run_worker", fake_run_worker)

        await worker.amain()

        assert calls == ["install", "run_worker"]
        assert event.is_set()

    asyncio.run(scenario())


def test_main_uses_asyncio_run(monkeypatch):
    worker = _reload_worker_module()
    executed = {}

    async def fake_amain() -> None:
        executed["amain"] = True

    def fake_run(coro):
        executed["asyncio_run"] = True
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(worker, "amain", fake_amain)
    monkeypatch.setattr(worker.asyncio, "run", fake_run)

    worker.main()

    assert executed == {"amain": True, "asyncio_run": True}


def test_models_qdrant_client_reexports(monkeypatch):
    dummy = ModuleType("app.qdrant_client")
    dummy.__all__ = ["QDRANT_URL", "search_chunks"]
    dummy.QDRANT_URL = "https://example-qdrant"
    dummy.search_chunks = object()

    monkeypatch.setitem(sys.modules, "app.qdrant_client", dummy)

    module = importlib.reload(importlib.import_module("app.models.qdrant_client"))

    assert module.QDRANT_URL == dummy.QDRANT_URL
    assert module.search_chunks is dummy.search_chunks


def test_rag_ingest_module_reexports(monkeypatch):
    dummy = ModuleType("app.ingest")
    dummy.__all__ = ["IngestService", "IngestWorker"]
    dummy.IngestService = object()
    dummy.IngestWorker = object()

    monkeypatch.setitem(sys.modules, "app.ingest", dummy)

    module = importlib.reload(importlib.import_module("app.rag.ingest"))

    assert module.IngestService is dummy.IngestService
    assert module.IngestWorker is dummy.IngestWorker

