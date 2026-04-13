"""Factory for constructing the FastAPI application."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import timezone
from importlib import import_module
from types import SimpleNamespace
from typing import Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
try:
    from starlette.middleware.base import BaseHTTPMiddleware
except ModuleNotFoundError:  # pragma: no cover - fallback for stubbed test envs
    class BaseHTTPMiddleware:  # type: ignore[override]
        """Fallback base class when Starlette middleware is unavailable."""

        def __init__(self, app):
            self.app = app


from app.api.error_responses import register_error_handlers

try:  # pragma: no cover - optional dependency resolution
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.schedulers.base import STATE_RUNNING
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    AsyncIOScheduler = None  # type: ignore[assignment]
    STATE_RUNNING = None  # type: ignore[assignment]
from app.chat.summarizer import ConversationSummarizer
try:
    from app.core.auth import TokenRegistry
except ImportError:  # pragma: no cover - fallback for heavily stubbed test modules
    class TokenRegistry:  # type: ignore[override]
        """Lightweight replacement used when the real auth module is unavailable."""

        def __init__(self) -> None:
            self._revoked: set[str] = set()
            self._inactive_users: set[str] = set()

        def revoke(self, token_id: str | None) -> None:
            if token_id:
                self._revoked.add(token_id)

        def is_revoked(self, token_id: str | None) -> bool:
            return bool(token_id and token_id in self._revoked)

        def mark_active(self, user_id: str | None) -> None:
            if user_id:
                self._inactive_users.discard(user_id)

        def mark_inactive(self, user_id: str | None) -> None:
            if user_id:
                self._inactive_users.add(user_id)

        def is_active(self, user_id: str | None) -> bool:
            return bool(user_id and user_id not in self._inactive_users)

from app.core.config import get_settings
from app.core.services import init_chat_store, init_memory_store
from app.ingest import IngestService, IngestWorker, parse_and_chunk  # noqa: F401
try:  # pragma: no cover - allow stubbed LLM modules in tests
    from app.llm import LLMProvider, get_cached_provider, reset_provider_cache
except ImportError:  # pragma: no cover - fallback when cache reset is unavailable
    from app.llm import LLMProvider, get_cached_provider  # type: ignore[assignment]

    def reset_provider_cache() -> None:  # type: ignore[redefining-outer-name]
        return None
try:  # pragma: no cover - allow heavily stubbed llm modules in tests
    from app.llm import lora_runtime
except ImportError:  # pragma: no cover - lightweight fallback when runtime missing
    def _lora_not_available(*_args, **_kwargs):
        raise RuntimeError("LoRA runtime is not available in this environment")


    lora_runtime = SimpleNamespace(  # type: ignore[assignment]
        load_adapter=_lora_not_available,
        unload_adapter=_lora_not_available,
        set_active_adapter=lambda *_args, **_kwargs: None,
        active_adapter=lambda: None,
        AdapterCompatibilityError=RuntimeError,
        RegistryError=RuntimeError,
    )
from app.llm.manager import LlamaLoraManager
from app.observability.metadata_guard import (
    check_sqlmodel_metadata,
    schedule_sqlmodel_metadata_guard,
)
from app.retriever import CrossEncoderReranker, get_reranker, get_vector_store
from app.services.files import FileStore, IngestQueue
from app.services import vectorstore as vectorstore_service
from app.ui import router as ui_router

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request identifier to the request state and response headers."""

    async def dispatch(self, request, call_next):  # type: ignore[override]
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request_id)
        return response


def _prepare_cors_origins(origins: Sequence[str] | None) -> list[str]:
    if not origins:
        return ["*"]

    cleaned: list[str] = []
    for origin in origins:
        value = (origin or "").strip()
        if value:
            cleaned.append(value)

    return cleaned or ["*"]


def _initialise_reranker(settings) -> CrossEncoderReranker | None:
    """Create a reranker instance when enabled in configuration."""

    if not getattr(settings, "rerank_enabled", False):
        return None

    try:  # pragma: no cover - optional dependency initialisation
        return get_reranker()
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to initialise cross-encoder reranker")
        return None


def _create_lora_manager(settings) -> LlamaLoraManager:
    """Instantiate the Lora manager, tolerating lightweight stubs in tests."""

    try:
        return LlamaLoraManager(settings)
    except TypeError as exc:  # pragma: no cover - compatibility shim for stubs
        try:
            return LlamaLoraManager()
        except TypeError:
            raise exc


def _scheduler_is_running(scheduler: object) -> bool:
    """Best-effort check for scheduler state across APScheduler versions."""

    running = getattr(scheduler, "running", None)
    if running is not None:
        return bool(running)

    state = getattr(scheduler, "state", None)
    if state is not None and STATE_RUNNING is not None:
        try:
            return state == STATE_RUNNING
        except Exception:  # pragma: no cover - defensive compatibility guard
            logger.debug("Unexpected scheduler state value: %r", state)

    return False


def create_app(provider: LLMProvider | None = None) -> FastAPI:
    """Build and configure the FastAPI application instance."""

    settings = get_settings()

    def setting(name: str, default):
        return getattr(settings, name, default)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker = getattr(app.state, "ingest_worker", None)
        service = getattr(app.state, "ingest_service", None)
        scheduler = getattr(app.state, "scheduler", None)
        worker_ready = worker is not None and service is not None and scheduler is not None
        if worker_ready:
            try:  # pragma: no cover - startup coordination
                service.ensure_background_worker()
                worker.ensure_started()
                app.state.ingest_worker_task = None
            except Exception:  # pragma: no cover - defensive startup logging
                logger.exception("Failed to start ingestion worker")
        if scheduler is not None and not _scheduler_is_running(scheduler):
            try:  # pragma: no cover - scheduler startup
                scheduler.start()
            except Exception:  # pragma: no cover - defensive scheduler logging
                logger.exception("Failed to start ingest scheduler")
        try:
            yield
        finally:
            service = getattr(app.state, "ingest_service", None)
            scheduler = getattr(app.state, "scheduler", None)
            if service is not None:
                stop_worker = getattr(service, "stop_background_worker", None)
                if callable(stop_worker):
                    try:  # pragma: no cover - shutdown coordination
                        await stop_worker()
                    except Exception:  # pragma: no cover - defensive shutdown logging
                        logger.exception("Failed to stop ingestion worker")
            if scheduler is not None and _scheduler_is_running(scheduler):
                try:  # pragma: no cover - scheduler shutdown
                    await scheduler.shutdown(wait=True)
                except Exception:  # pragma: no cover - defensive scheduler logging
                    logger.exception("Failed to shut down scheduler")

    try:
        application = FastAPI(title="kb", lifespan=lifespan)
    except TypeError:
        application = FastAPI(title="kb")

        async def _startup_lifespan() -> None:
            context = lifespan(application)
            application.state._kb_lifespan_context = context  # type: ignore[attr-defined]
            await context.__aenter__()

        async def _shutdown_lifespan() -> None:
            context = getattr(application.state, "_kb_lifespan_context", None)
            if context is None:
                return
            await context.__aexit__(None, None, None)

        application.on_event("startup")(_startup_lifespan)
        application.on_event("shutdown")(_shutdown_lifespan)

    cors_origins = _prepare_cors_origins(setting("cors_allow_origins", None))
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
        expose_headers=["*"],
    )
    application.add_middleware(RequestIDMiddleware)

    register_error_handlers(application)

    chat_store = init_chat_store(settings)
    memory_store = init_memory_store(settings)
    reset_provider_cache()
    llm_provider = provider or get_cached_provider(settings)
    vector_store = get_vector_store(settings)
    lora_manager = _create_lora_manager(settings)
    reranker = _initialise_reranker(settings)
    summarizer = ConversationSummarizer(chat_store, llm_provider.generate)

    file_store = FileStore()
    ingest_queue = IngestQueue()
    scheduler = None
    if AsyncIOScheduler is None:  # pragma: no cover - optional dependency guard
        logger.warning(
            "APScheduler is not installed; background scheduling is disabled"
        )
    else:
        scheduler = AsyncIOScheduler(timezone=timezone.utc)
        schedule_sqlmodel_metadata_guard(scheduler)

    ingest_autostart = bool(setting("ingest_autostart_worker", True))
    ingest_use_local_queue = bool(setting("ingest_use_local_queue", True))
    ingest_service = IngestService(
        max_retries=setting("ingest_max_retries", 3),
        backoff_seconds=setting("ingest_backoff_seconds", 1.0),
        auto_process=ingest_autostart,
        use_local_queue=ingest_use_local_queue,
    )
    ingest_worker = None
    if ingest_autostart:
        ingest_worker = IngestWorker(ingest_service)
        set_worker = getattr(ingest_service, "set_worker", None)
        if callable(set_worker):
            set_worker(ingest_worker)
        configure_scheduler = getattr(ingest_service, "configure_scheduler", None)
        if callable(configure_scheduler) and scheduler is not None:
            configure_scheduler(scheduler)

    min_citations, max_citations = setting("citations_bounds", (3, 5))

    application.state.settings = settings
    application.state.chat_store = chat_store
    application.state.llm_provider = llm_provider
    application.state.llm_client = llm_provider
    application.state.lora_manager = lora_manager
    application.state.vector_store = vector_store
    application.state.memory_store = memory_store
    application.state.memory_store_factory = init_memory_store
    application.state.file_store = file_store
    application.state.ingest_queue = ingest_queue
    application.state.ingest_service = ingest_service
    application.state.ingest_worker = ingest_worker
    application.state.ingest_worker_task = None
    application.state.scheduler = scheduler
    application.state.summarizer = summarizer
    application.state.reranker = reranker
    fallback_index: list[dict[str, object]] = []
    vectorstore_service.set_fallback_storage(fallback_index)
    application.state.fallback_index = fallback_index
    application.state.chat_history_limit = setting("chat_history_limit", 12)
    application.state.retrieve_topk = setting("retrieve_topk", 10)
    application.state.rerank_topk = setting("rerank_topk", 50)
    application.state.rerank_enabled = setting("rerank_enabled", True)
    application.state.chat_summary_trigger = setting("chat_summary_trigger", 10)
    application.state.min_citations = min_citations
    application.state.max_citations = max_citations
    application.state.token_registry = TokenRegistry()

    if getattr(settings, "use_lora", False):
        default_adapter = getattr(settings, "lora_default_adapter", "none") or "none"
        if default_adapter.lower() not in {"none", "0", "false", "no"}:
            try:
                lora_runtime.load_adapter(default_adapter)
            except Exception:  # pragma: no cover - best effort startup log
                logger.exception("Failed to load default LoRA adapter: %s", default_adapter)

    application.include_router(ui_router)

    api_router_module = import_module("app.api.router")
    application.include_router(getattr(api_router_module, "api_router"))

    # Prime the metadata guard once during application construction so metrics are
    # populated even before the first scheduler tick.
    check_sqlmodel_metadata(origin="startup")

    return application


__all__ = ["create_app"]
