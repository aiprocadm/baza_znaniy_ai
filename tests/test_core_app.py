import importlib
import sys
import types

import pytest
from fastapi import APIRouter
from importlib.machinery import ModuleSpec


@pytest.fixture
def core_app(monkeypatch):
    """Import ``app.core.app`` with router dependencies stubbed out."""

    for name in [
        "app.core.app",
        "app.core.config",
        "app.core.services",
        "app.api",
        "app.api.router",
        "app.api.routes",
        "app.ingest",
        "app.ingest.service",
        "app.services",
        "app.services.vectorstore",
        "app.retriever",
        "app.chat",
        "app.chat.summarizer",
        "app.llm",
        "app.llm.manager",
        "app.models.lora",
        "app.services.files",
    ]:
        sys.modules.pop(name, None)

    config_module = types.ModuleType("app.core.config")
    config_module.__spec__ = ModuleSpec("app.core.config", loader=None)

    class StubSettings:  # pragma: no cover - simple stub
        secret_key = "stub-secret"
        jwt_algorithm = "HS256"
        access_token_expire_minutes = 15

    config_module.Settings = StubSettings
    config_module.get_settings = lambda: StubSettings()
    monkeypatch.setitem(sys.modules, "app.core.config", config_module)

    core_services_module = types.ModuleType("app.core.services")
    core_services_module.__spec__ = ModuleSpec("app.core.services", loader=None)
    core_services_module.init_chat_store = lambda settings: object()
    core_services_module.init_memory_store = lambda settings: object()
    monkeypatch.setitem(sys.modules, "app.core.services", core_services_module)

    retriever_module = types.ModuleType("app.retriever")
    retriever_module.__spec__ = ModuleSpec("app.retriever", loader=None)
    retriever_module.CrossEncoderReranker = type("CrossEncoderReranker", (), {})
    retriever_module.get_reranker = lambda: retriever_module.CrossEncoderReranker()
    retriever_module.get_vector_store = lambda settings=None: object()
    monkeypatch.setitem(sys.modules, "app.retriever", retriever_module)

    ingest_module = types.ModuleType("app.ingest")
    ingest_module.__spec__ = ModuleSpec("app.ingest", loader=None)

    class StubIngestService:  # noqa: D401 - simple stub for tests
        """Stub ingest service used for isolated tests."""

        def __init__(self, *args, **kwargs):  # pragma: no cover - simple stub
            pass

    class StubIngestWorker:
        def __init__(self, *args, **kwargs):  # pragma: no cover - simple stub
            pass

    def stub_parse_and_chunk(*args, **kwargs):  # pragma: no cover - simple stub
        return []

    ingest_module.IngestService = StubIngestService
    ingest_module.IngestWorker = StubIngestWorker
    ingest_module.parse_and_chunk = stub_parse_and_chunk
    monkeypatch.setitem(sys.modules, "app.ingest", ingest_module)

    ingest_service_module = types.ModuleType("app.ingest.service")
    ingest_service_module.IngestService = StubIngestService
    ingest_service_module.IngestWorker = StubIngestWorker
    ingest_service_module.parse_and_chunk = stub_parse_and_chunk
    monkeypatch.setitem(sys.modules, "app.ingest.service", ingest_service_module)

    services_module = types.ModuleType("app.services")
    services_module.__path__ = []  # mark as package for submodule imports
    services_module.__spec__ = ModuleSpec("app.services", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "app.services", services_module)

    vectorstore_module = types.ModuleType("app.services.vectorstore")
    vectorstore_module.__spec__ = ModuleSpec("app.services.vectorstore", loader=None)
    vectorstore_module.get_vector_store = lambda *args, **kwargs: object()
    vectorstore_module.index_chunks = lambda *args, **kwargs: None
    vectorstore_module.set_fallback_storage = lambda storage: None
    vectorstore_module.get_fallback_storage = lambda: []
    vectorstore_module.clear_fallback = lambda: None
    monkeypatch.setitem(sys.modules, "app.services.vectorstore", vectorstore_module)

    files_module = types.ModuleType("app.services.files")
    files_module.__spec__ = ModuleSpec("app.services.files", loader=None)

    class StubFileStore:
        pass

    class StubIngestQueue:
        pass

    files_module.FileStore = StubFileStore
    files_module.IngestQueue = StubIngestQueue
    monkeypatch.setitem(sys.modules, "app.services.files", files_module)

    chat_module = types.ModuleType("app.chat")
    chat_module.__path__ = []
    chat_module.__spec__ = ModuleSpec("app.chat", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "app.chat", chat_module)

    summarizer_module = types.ModuleType("app.chat.summarizer")
    summarizer_module.__spec__ = ModuleSpec("app.chat.summarizer", loader=None)

    class StubSummarizer:
        def __init__(self, *args, **kwargs):  # pragma: no cover - simple stub
            pass

    summarizer_module.ConversationSummarizer = StubSummarizer
    monkeypatch.setitem(sys.modules, "app.chat.summarizer", summarizer_module)

    llm_module = types.ModuleType("app.llm")
    llm_module.__spec__ = ModuleSpec("app.llm", loader=None, is_package=True)

    class DummyProvider:
        def __init__(self):
            self.generate = lambda *args, **kwargs: None

    llm_module.LLMProvider = DummyProvider
    llm_module.get_cached_provider = lambda settings: DummyProvider()
    monkeypatch.setitem(sys.modules, "app.llm", llm_module)

    llm_providers_module = types.ModuleType("app.llm.providers")
    llm_providers_module.__spec__ = ModuleSpec("app.llm.providers", loader=None)
    llm_providers_module.LLMProvider = DummyProvider
    llm_providers_module.get_llm_provider = lambda *args, **kwargs: DummyProvider()
    monkeypatch.setitem(sys.modules, "app.llm.providers", llm_providers_module)

    llm_manager_module = types.ModuleType("app.llm.manager")
    llm_manager_module.__spec__ = ModuleSpec("app.llm.manager", loader=None)
    llm_manager_module.LlamaLoraManager = type("LlamaLoraManager", (), {})
    monkeypatch.setitem(sys.modules, "app.llm.manager", llm_manager_module)

    models_lora_module = types.ModuleType("app.models.lora")
    models_lora_module.__spec__ = ModuleSpec("app.models.lora", loader=None)

    class StubLoraStatusResponse:
        @staticmethod
        def from_status(status):  # pragma: no cover - simple stub
            return types.SimpleNamespace(model_dump=lambda: {})

    models_lora_module.LoraStatusResponse = StubLoraStatusResponse

    class StubLoraLoadRequest:
        def __init__(self, **data):  # pragma: no cover - simple stub
            self.__dict__.update(data)

        def model_dump(self, *args, **kwargs):  # pragma: no cover - simple stub
            return dict(self.__dict__)

    class StubLoraUnloadRequest(StubLoraLoadRequest):  # pragma: no cover - simple stub
        pass

    models_lora_module.LoraLoadRequest = StubLoraLoadRequest
    models_lora_module.LoraUnloadRequest = StubLoraUnloadRequest
    monkeypatch.setitem(sys.modules, "app.models.lora", models_lora_module)

    router_module = types.ModuleType("app.api.router")
    router_module.__spec__ = ModuleSpec("app.api.router", loader=None)
    router_module.api_router = APIRouter()
    monkeypatch.setitem(sys.modules, "app.api.router", router_module)

    api_module = types.ModuleType("app.api")
    api_module.__spec__ = ModuleSpec("app.api", loader=None, is_package=True)
    api_module.api_router = router_module.api_router
    api_module.router = router_module
    monkeypatch.setitem(sys.modules, "app.api", api_module)
    monkeypatch.setitem(sys.modules, "app.api.routes", types.ModuleType("app.api.routes"))

    return importlib.import_module("app.core.app")


class DummySettings:
    def __init__(
        self,
        cors_allow_origins,
        rerank_enabled,
        ingest_max_retries=1,
        ingest_backoff_seconds=0.1,
        chat_history_limit=5,
        retrieve_topk=10,
        rerank_topk=4,
        chat_summary_trigger=2,
        citations_bounds=(1, 3),
    ) -> None:
        self.cors_allow_origins = list(cors_allow_origins)
        self.rerank_enabled = rerank_enabled
        self.ingest_max_retries = ingest_max_retries
        self.ingest_backoff_seconds = ingest_backoff_seconds
        self.chat_history_limit = chat_history_limit
        self.retrieve_topk = retrieve_topk
        self.rerank_topk = rerank_topk
        self.chat_summary_trigger = chat_summary_trigger
        self._citations_bounds = citations_bounds

    @property
    def citations_bounds(self) -> tuple[int, int]:
        return self._citations_bounds


def test_prepare_cors_origins_with_none_returns_wildcard(core_app):
    assert core_app._prepare_cors_origins(None) == ["*"]


def test_prepare_cors_origins_trims_and_ignores_empty_values(core_app):
    assert core_app._prepare_cors_origins(["", "   ", "\n"]) == ["*"]


def test_prepare_cors_origins_returns_cleaned_values(core_app):
    result = core_app._prepare_cors_origins([
        " https://example.com ",
        "",
        "https://foo.test",
    ])

    assert result == ["https://example.com", "https://foo.test"]


def test_v1_router_registers_admin_routes(monkeypatch):
    import importlib
    import sys
    import types

    from fastapi import APIRouter
    from pathlib import Path

    for name in list(sys.modules):
        if name == "app.api.v1" or name.startswith("app.api.v1."):
            sys.modules.pop(name, None)

    api_package = types.ModuleType("app.api")
    api_package.__path__ = [str(Path("app/api"))]
    api_package.__spec__ = ModuleSpec("app.api", loader=None, is_package=True)
    monkeypatch.setitem(sys.modules, "app.api", api_package)

    module_prefixes = {
        "admin": "/admin",
        "auth": "/auth",
        "users": "/users",
        "tenants": "/tenants",
        "upload": "/upload",
        "ingest": "/ingest",
        "search": "/search",
        "chat": "/chat",
        "files": "/files",
        "delete": "/delete",
        "lora": "/lora",
    }

    for module_name, prefix in module_prefixes.items():
        module = types.ModuleType(f"app.api.v1.{module_name}")
        module.__spec__ = ModuleSpec(module.__name__, loader=None)
        module.router = APIRouter(prefix=prefix)
        module.router.get("/ping")(lambda module_name=module_name: module_name)
        monkeypatch.setitem(sys.modules, module.__name__, module)

    v1_module = importlib.import_module("app.api.v1")
    router = v1_module.router
    sys.modules.pop("app.api.v1", None)

    routes = getattr(router, "routes", getattr(router, "_routes", []))
    paths = {
        getattr(route, "path", getattr(route, "path_format", None))
        for route in routes
    }
    assert "/admin/ping" in paths


def test_initialise_reranker_disabled(core_app, monkeypatch):
    monkeypatch.setattr(
        core_app,
        "get_reranker",
        lambda: pytest.fail("reranker should not be initialised when disabled"),
    )

    settings = types.SimpleNamespace(rerank_enabled=False)

    assert core_app._initialise_reranker(settings) is None


def test_initialise_reranker_returns_stub_when_enabled(core_app, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(core_app, "get_reranker", lambda: sentinel)

    settings = types.SimpleNamespace(rerank_enabled=True)

    assert core_app._initialise_reranker(settings) is sentinel


def test_initialise_reranker_logs_failure(core_app, monkeypatch, caplog):
    def broken_reranker():
        raise RuntimeError("boom")

    monkeypatch.setattr(core_app, "get_reranker", broken_reranker)

    settings = types.SimpleNamespace(rerank_enabled=True)

    with caplog.at_level("ERROR"):
        result = core_app._initialise_reranker(settings)

    assert result is None
    assert any(
        "Failed to initialise cross-encoder reranker" in message
        for message in caplog.messages
    )


def _stub_app_dependencies(core_app, monkeypatch):
    chat_store = object()
    memory_store = object()
    vector_store = object()
    lora_manager = object()
    summarizer = object()
    file_store = object()
    ingest_queue = object()

    provider = types.SimpleNamespace(generate=lambda *args, **kwargs: None)

    monkeypatch.setattr(core_app, "init_chat_store", lambda settings: chat_store)
    monkeypatch.setattr(core_app, "init_memory_store", lambda settings: memory_store)
    monkeypatch.setattr(core_app, "get_cached_provider", lambda settings: provider)
    monkeypatch.setattr(core_app, "get_vector_store", lambda settings: vector_store)
    monkeypatch.setattr(core_app, "LlamaLoraManager", lambda settings: lora_manager)
    monkeypatch.setattr(core_app, "ConversationSummarizer", lambda *args, **kwargs: summarizer)
    monkeypatch.setattr(core_app, "FileStore", lambda: file_store)
    monkeypatch.setattr(core_app, "IngestQueue", lambda: ingest_queue)

    class DummyIngestService:
        def __init__(
            self,
            max_retries,
            backoff_seconds,
            auto_process=False,
            use_local_queue=True,
        ):
            self.max_retries = max_retries
            self.backoff_seconds = backoff_seconds
            self.auto_process = auto_process
            self.use_local_queue = use_local_queue

        def set_worker(self, worker):  # pragma: no cover - simple stub
            self.worker = worker

    class DummyIngestWorker:
        def __init__(self, service):
            self.service = service

    monkeypatch.setattr(core_app, "IngestService", DummyIngestService)
    monkeypatch.setattr(core_app, "IngestWorker", DummyIngestWorker)

    return {
        "chat_store": chat_store,
        "memory_store": memory_store,
        "provider": provider,
        "vector_store": vector_store,
        "lora_manager": lora_manager,
        "summarizer": summarizer,
        "file_store": file_store,
        "ingest_queue": ingest_queue,
    }


def test_create_app_normalises_cors_and_injects_reranker(core_app, monkeypatch):
    dependencies = _stub_app_dependencies(core_app, monkeypatch)

    settings = DummySettings(
        cors_allow_origins=[" https://one.test ", "https://two.test  "],
        rerank_enabled=True,
    )
    monkeypatch.setattr(core_app, "get_settings", lambda: settings)

    captured = {}
    cors_options: dict[str, object] = {}

    original_add_middleware = core_app.FastAPI.add_middleware

    def recording_add_middleware(self, middleware_class, **options):
        if middleware_class is core_app.CORSMiddleware:
            cors_options.update(options)
        return original_add_middleware(self, middleware_class, **options)

    monkeypatch.setattr(core_app.FastAPI, "add_middleware", recording_add_middleware)

    def fake_initialise(settings_arg):
        captured["settings"] = settings_arg
        return "reranker"

    monkeypatch.setattr(core_app, "_initialise_reranker", fake_initialise)

    application = core_app.create_app()

    assert captured["settings"] is settings
    assert application.state.reranker == "reranker"
    assert application.state.chat_store is dependencies["chat_store"]
    assert application.state.llm_provider is dependencies["provider"]

    assert cors_options["allow_origins"] == [
        "https://one.test",
        "https://two.test",
    ]


def test_create_app_excludes_reranker_when_disabled(core_app, monkeypatch):
    _stub_app_dependencies(core_app, monkeypatch)

    settings = DummySettings(cors_allow_origins=["*"], rerank_enabled=False)
    monkeypatch.setattr(core_app, "get_settings", lambda: settings)

    monkeypatch.setattr(core_app, "_initialise_reranker", lambda settings_arg: None)

    application = core_app.create_app()

    assert application.state.reranker is None
    assert application.state.rerank_enabled is False
