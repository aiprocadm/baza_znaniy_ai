import io
import sys
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from app.api import routes as routes_module


class LegacyProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ensure_model(self) -> None:
        self.calls.append("model")

    def generate(self, prompt: str) -> str | None:
        self.calls.append(prompt)
        return None


class FullProvider(LegacyProvider):
    def ensure_ready(self) -> None:  # pragma: no cover - passthrough
        self.calls.append("ready")


class LegacyChatStore:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []


class ExplodingVectorStore:
    def ensure_ready(self) -> None:
        raise RuntimeError("offline")

    def upsert(self, items):  # pragma: no cover - defensive path
        raise AssertionError("upsert should not be called when ensure_ready fails")


@pytest.fixture(autouse=True)
def reset_settings_cache():
    from app.core import config as config_module

    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


def test_coerce_llm_provider_wraps_missing_lifecycle_hooks():
    legacy = LegacyProvider()
    wrapped = routes_module._coerce_llm_provider(legacy)

    assert wrapped is not legacy
    wrapped.ensure_model()
    wrapped.ensure_ready()
    text = wrapped.generate("hello")

    assert text == "Ответ"
    assert legacy.calls == ["model", "hello"]


def test_coerce_llm_provider_returns_original_when_complete():
    provider = FullProvider()
    wrapped = routes_module._coerce_llm_provider(provider)

    assert wrapped is provider


def test_coerce_chat_store_adapts_missing_methods():
    adapter = routes_module._coerce_chat_store(LegacyChatStore())
    conversation_id = adapter.ensure_conversation("user", None)

    adapter.record_exchange(conversation_id, "question", "answer")

    assert adapter.get_recent_messages(conversation_id, limit=5) == [
        ("user", "question"),
        ("assistant", "answer"),
    ]
    assert adapter.get_summary(conversation_id) == ""
    assert adapter.messages_since_summary(conversation_id) == 1


def test_vector_store_adapter_falls_back_to_index(monkeypatch):
    fallback_index: list[dict[str, int]] = [{"file": 1, "page": 1}]
    adapter = routes_module._coerce_vector_store(ExplodingVectorStore(), fallback_index)

    adapter.ensure_ready()
    hits = adapter.search("query", top_k=1)
    assert hits == fallback_index


def test_vector_store_adapter_produces_synthetic_hits_when_empty():
    adapter = routes_module._coerce_vector_store(ExplodingVectorStore(), [])

    hits = adapter.search("question", top_k=2)
    assert hits
    assert all("file" in hit for hit in hits)


def test_memory_store_created_via_module_factory(monkeypatch, tmp_path):
    class DummyStore:
        def __init__(
            self, *, db_path: str, ttl_days: int, summary_trigger: int, max_tokens: int
        ) -> None:
            self.db_path = db_path
            self.ttl_days = ttl_days
            self.summary_trigger = summary_trigger
            self.max_tokens = max_tokens

        def load_context(self, *args, **kwargs):  # pragma: no cover - adapter behaviour
            return {}

        def record(self, *args, **kwargs):  # pragma: no cover - adapter behaviour
            return None

    def init_helper(settings):
        return DummyStore(
            db_path=str(tmp_path / "memory.db"),
            ttl_days=settings.chat_memory_ttl_days,
            summary_trigger=settings.chat_summary_trigger,
            max_tokens=settings.chat_memory_max_tokens,
        )

    module_name = "kb_service_app.main"
    sys.modules[module_name] = SimpleNamespace(
        MemoryStore=DummyStore,
        _init_memory_store=init_helper,
    )

    state = SimpleNamespace(memory_store=None, memory_store_factory=lambda *_: object())
    settings = SimpleNamespace(
        chat_memory_enabled=True,
        chat_memory_ttl_days=3,
        chat_summary_trigger=5,
        chat_memory_max_tokens=512,
        data_dir=tmp_path,
        memory_db_path_resolved=tmp_path / "configured.db",
    )

    try:
        store = routes_module._ensure_memory_store(state, settings)
    finally:
        sys.modules.pop(module_name, None)

    assert isinstance(store, DummyStore)
    assert state.memory_store is store
    assert store.ttl_days == 3
    assert store.max_tokens == 512


def test_memory_store_disabled_returns_none(monkeypatch):
    state = SimpleNamespace(memory_store=None)
    settings = SimpleNamespace(chat_memory_enabled=False)

    assert routes_module._ensure_memory_store(state, settings) is None


def test_content_length_check_honours_limits():
    request = SimpleNamespace(headers={"content-length": "1025"})
    limits = SimpleNamespace(max_bytes=1024)

    assert routes_module._content_length_exceeds_limits(request, limits) is True


def test_coerce_upload_file_handles_nested_tuples():
    upload = routes_module._coerce_upload_file(
        [
            ("doc.txt", b"data", "text/plain"),
        ]
    )

    assert isinstance(upload, UploadFile)
    assert upload.filename == "doc.txt"
    assert upload.content_type == "text/plain"
    assert upload.file.read() == b"data"


def test_coerce_bytes_resets_stream_position():
    stream = io.BytesIO(b"abcdef")
    stream.seek(3)

    data = routes_module._coerce_bytes(stream)

    assert data == b"abcdef"
    assert stream.tell() == 3


def test_index_chunks_uses_fallback_on_failure(monkeypatch):
    fallback_index: list[dict[str, str]] = []
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                vector_store=ExplodingVectorStore(), fallback_index=fallback_index
            )
        )
    )
    recorded: list[tuple[str, str, int]] = []

    def record(operation, backend, count, duration):
        recorded.append((operation, backend, count))

    monkeypatch.setattr(routes_module, "record_index_operation", record)

    processed = routes_module._index_chunks(request, [{"id": 1}])

    assert processed == 1
    assert fallback_index == [{"id": 1}]
    assert ("error", "vector", 1) in recorded
    assert ("success", "fallback", 1) in recorded
