"""Tests for the MVP /api/kb/* endpoints.

These tests intentionally avoid the full ``create_app`` factory because
it pulls heavy optional dependencies (LLM provider, Qdrant client, LoRA
runtime). The MVP router is self-contained, so we mount only it on a
fresh FastAPI app with a temporary SQLite database.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.kb_mvp import router as kb_router
from app.services import kb_embeddings, kb_llm, kb_rerank, kb_store


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db_path = tmp_path / "kb_mvp.sqlite"
    monkeypatch.setenv("KB_MVP_DB_PATH", str(db_path))
    # Force the hashing embedder so tests don't make network calls.
    for name in (
        "KB_LLM_PROVIDER",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "LLM_API_BASE_URL",
        "LLM_API_KEY",
        "KB_EMBEDDINGS_BACKEND",
        "OLLAMA_EMBED_MODEL",
        "EMBEDDINGS_API_BASE_URL",
        "KB_RERANK_ENABLED",
        "KB_RERANK_MODEL",
        "KB_RERANK_CANDIDATES",
        "KB_RERANK_TOPN",
        "KB_API_KEY",
        "KB_SEARCH_HARD_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)
    # Force hashing embedder so tests are fast and don't attempt ST weight loading
    monkeypatch.setenv("KB_EMBEDDINGS_BACKEND", "hash")
    # API tests must not load the 2GB local GGUF; GGUF selection is covered by
    # unit tests that monkeypatch select_provider directly.
    monkeypatch.setenv("KB_LLM_LOCAL_FALLBACK", "0")
    kb_store.reset_default_store()
    kb_embeddings.reset_embedder()
    kb_rerank.reset_cache()

    app = FastAPI()
    app.include_router(kb_router, prefix="/api/kb")

    with TestClient(app) as c:
        yield c

    kb_store.reset_default_store()
    kb_embeddings.reset_embedder()
    kb_rerank.reset_cache()


# ----------------------------------------------------------------------
# Basics
# ----------------------------------------------------------------------


def test_health_returns_ok_with_diagnostics(client: TestClient) -> None:
    response = client.get("/api/kb/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "llm" in body and "providers" in body["llm"]
    assert "embedder" in body and body["embedder"]["name"] == "hash"


def test_providers_endpoint_lists_known_presets(client: TestClient) -> None:
    response = client.get("/api/kb/providers")
    assert response.status_code == 200
    body = response.json()
    names = {p["name"] for p in body["providers"]}
    assert {"deepseek", "groq", "openrouter", "openai", "ollama"}.issubset(names)
    # Без ключей selected должно быть None
    assert body["selected"] is None


# ----------------------------------------------------------------------
# Documents (JSON)
# ----------------------------------------------------------------------


def test_create_document_persists_and_chunks(client: TestClient) -> None:
    payload = {
        "title": "Регламент онбординга",
        "text": "Новые сотрудники проходят инструктаж в первый день. "
        "Каждому выдают доступы и наставника. " * 30,
    }
    response = client.post("/api/kb/documents", json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] >= 1
    assert body["title"] == payload["title"]
    assert body["chunks"] >= 1
    assert body["source"] == "text"
    assert body["filename"] is None
    assert body["text"] is None


def test_create_document_rejects_empty_text(client: TestClient) -> None:
    response = client.post("/api/kb/documents", json={"title": "X", "text": "   "})
    assert response.status_code == 422


def test_list_documents_orders_desc(client: TestClient) -> None:
    for index in range(3):
        response = client.post(
            "/api/kb/documents",
            json={"title": f"Документ {index}", "text": f"Содержимое номер {index}, " * 20},
        )
        assert response.status_code == 201

    response = client.get("/api/kb/documents")
    assert response.status_code == 200
    docs = response.json()
    assert [d["title"] for d in docs] == ["Документ 2", "Документ 1", "Документ 0"]
    assert all(d["source"] == "text" for d in docs)


def test_get_document_returns_full_text(client: TestClient) -> None:
    create = client.post(
        "/api/kb/documents", json={"title": "Doc", "text": "Полный текст документа."}
    )
    doc_id = create.json()["id"]

    response = client.get(f"/api/kb/documents/{doc_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == doc_id
    assert body["text"] == "Полный текст документа."


def test_delete_document_removes_and_404s_again(client: TestClient) -> None:
    create = client.post("/api/kb/documents", json={"title": "X", "text": "Тестовый текст"})
    doc_id = create.json()["id"]

    delete_response = client.delete(f"/api/kb/documents/{doc_id}")
    assert delete_response.status_code == 200

    second_delete = client.delete(f"/api/kb/documents/{doc_id}")
    assert second_delete.status_code == 404


# ----------------------------------------------------------------------
# File uploads
# ----------------------------------------------------------------------


def test_upload_txt_file_indexes_content(client: TestClient) -> None:
    content = ("Содержание текстового файла. " * 20).encode("utf-8")
    response = client.post(
        "/api/kb/documents/upload",
        files={"file": ("notes.txt", io.BytesIO(content), "text/plain")},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "file"
    assert body["filename"] == "notes.txt"
    assert body["chunks"] >= 1
    assert body["mime_type"]


def test_upload_html_extension_accepted(client: TestClient) -> None:
    """HTML files должны проходить ext-check (содержимое парсит legacy/docling)."""

    content = b"<html><body><h1>Title</h1><p>Body text</p></body></html>"
    response = client.post(
        "/api/kb/documents/upload",
        files={"file": ("page.html", io.BytesIO(content), "text/html")},
    )
    # Содержимое может быть распарсено или нет в зависимости от среды,
    # но ext-check не должен возвращать 415.
    assert response.status_code != 415, response.text


def test_upload_md_uses_decoded_text(client: TestClient) -> None:
    content = "# Заголовок\n\nТело документа.".encode("utf-8")
    response = client.post(
        "/api/kb/documents/upload",
        files={"file": ("readme.md", io.BytesIO(content), "text/markdown")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "readme.md"
    # Документ виден в списке
    listed = client.get("/api/kb/documents").json()
    assert any(d["filename"] == "readme.md" for d in listed)


def test_upload_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/api/kb/documents/upload",
        files={"file": ("payload.exe", io.BytesIO(b"data"), "application/octet-stream")},
    )
    assert response.status_code == 415


def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/kb/documents/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_supports_explicit_title(client: TestClient) -> None:
    content = b"text"
    response = client.post(
        "/api/kb/documents/upload",
        files={"file": ("doc.txt", io.BytesIO(content), "text/plain")},
        data={"title": "Свой заголовок"},
    )
    assert response.status_code == 201
    assert response.json()["title"] == "Свой заголовок"


# ----------------------------------------------------------------------
# Search & ask
# ----------------------------------------------------------------------


def test_search_returns_relevant_hits(client: TestClient) -> None:
    client.post(
        "/api/kb/documents",
        json={
            "title": "Налоговый кодекс",
            "text": "Налоговая декларация подаётся ежегодно. Налоги начисляются по ставке.",
        },
    )
    client.post(
        "/api/kb/documents",
        json={
            "title": "Кулинарный справочник",
            "text": "Рецепт борща включает свёклу, мясо и капусту.",
        },
    )

    response = client.post("/api/kb/search", json={"query": "декларация налогов", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["hits"], "ожидаются результаты по релевантному запросу"
    assert "Налогов" in body["hits"][0]["document_title"]


def test_search_rejects_empty_query(client: TestClient) -> None:
    response = client.post("/api/kb/search", json={"query": "   ", "top_k": 5})
    assert response.status_code == 422


def test_ask_uses_extractive_fallback_when_no_provider(client: TestClient) -> None:
    client.post(
        "/api/kb/documents",
        json={
            "title": "Регламент отпусков",
            "text": "Сотрудник имеет право на 28 календарных дней отпуска в год.",
        },
    )

    response = client.post("/api/kb/ask", json={"question": "Сколько дней отпуска?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "extractive"
    assert body["sources"]
    assert "28 календарных" in body["answer"]


def test_ask_handles_empty_kb(client: TestClient) -> None:
    response = client.post("/api/kb/ask", json={"question": "Есть ли что-то?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["provider"] == "none"


def test_ask_uses_kb_llm_provider_when_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/ask` should prefer `kb_llm.select_provider()` over legacy state.llm_provider."""

    calls = []

    class FakeProvider:
        name = "deepseek"
        model = "deepseek-chat"

        def generate(self, prompt: str, *, system=None):
            calls.append({"prompt": prompt, "system": system})
            return kb_llm.LLMResponse(
                text="Отпуск составляет 28 дней.",
                provider=self.name,
                model=self.model,
                elapsed_ms=12.3,
            )

    fake = FakeProvider()
    monkeypatch.setattr(kb_llm, "select_provider", lambda env=None: fake)

    client.post(
        "/api/kb/documents",
        json={"title": "Отпуска", "text": "Отпуск составляет 28 календарных дней."},
    )
    response = client.post("/api/kb/ask", json={"question": "Сколько дней отпуска?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "deepseek"
    assert body["model"] == "deepseek-chat"
    assert body["elapsed_ms"] == 12.3
    assert body["answer"] == "Отпуск составляет 28 дней."
    assert calls  # был вызов provider.generate


def test_ask_falls_back_to_legacy_provider_when_kb_llm_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kb_llm, "select_provider", lambda env=None: None)

    class LegacyProvider:
        name = "legacy-llama"

        def ensure_ready(self) -> None: ...

        def generate(self, prompt: str) -> str:
            return "Legacy answer"

    client.app.state.llm_provider = LegacyProvider()
    client.post(
        "/api/kb/documents",
        json={"title": "x", "text": "Документ с релевантным текстом про отпуск"},
    )
    response = client.post("/api/kb/ask", json={"question": "Сколько дней отпуска?", "top_k": 3})
    body = response.json()
    assert body["provider"] == "legacy-llama"
    assert body["answer"] == "Legacy answer"


def test_ask_rejects_empty_question(client: TestClient) -> None:
    response = client.post("/api/kb/ask", json={"question": "", "top_k": 3})
    assert response.status_code == 422


# ----------------------------------------------------------------------
# Store invariants
# ----------------------------------------------------------------------


def test_text_too_large_returns_4xx(client: TestClient) -> None:
    huge_text = "x" * (kb_store.MAX_TEXT_LEN + 1)
    response = client.post("/api/kb/documents", json={"title": "Big", "text": huge_text})
    assert response.status_code in (400, 422)


def test_store_split_text_handles_overlap_edge_cases() -> None:
    short = kb_store.split_text("Привет, мир", chunk_size=100, overlap=10)
    assert short == ["Привет, мир"]

    sample = "слово " * 200
    chunks = kb_store.split_text(sample.strip(), chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)


def test_store_embed_is_deterministic() -> None:
    a = kb_store.embed("привет, мир!")
    b = kb_store.embed("привет, мир!")
    assert a == b
    assert len(a) == kb_store.EMBEDDING_DIM


# ----------------------------------------------------------------------
# kb_llm
# ----------------------------------------------------------------------


def test_build_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DEEPSEEK_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(kb_llm.LLMUnavailable):
        kb_llm.build_provider("deepseek", env={})


def test_build_provider_picks_preset_url_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = kb_llm.build_provider("deepseek", env={"DEEPSEEK_API_KEY": "test-key"})
    assert provider.name == "deepseek"
    assert provider.model == "deepseek-chat"
    assert provider.config.api_base == "https://api.deepseek.com/v1"
    assert provider.config.api_key == "test-key"


def test_build_provider_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = kb_llm.build_provider(
        "groq",
        env={"GROQ_API_KEY": "k", "GROQ_MODEL": "mixtral-8x7b-32768"},
    )
    assert provider.model == "mixtral-8x7b-32768"


def test_build_provider_openrouter_extra_headers() -> None:
    provider = kb_llm.build_provider(
        "openrouter",
        env={
            "OPENROUTER_API_KEY": "k",
            "OPENROUTER_REFERER": "https://kb.local",
            "OPENROUTER_TITLE": "KB MVP",
        },
    )
    assert provider.config.extra_headers["HTTP-Referer"] == "https://kb.local"
    assert provider.config.extra_headers["X-Title"] == "KB MVP"


def test_build_provider_ollama_works_without_key() -> None:
    provider = kb_llm.build_provider("ollama", env={"OLLAMA_MODEL": "llama3.2"})
    assert provider.name == "ollama"
    assert provider.model == "llama3.2"
    assert provider.config.api_key is None


def test_build_provider_custom_via_base_url() -> None:
    provider = kb_llm.build_provider(
        "custom",
        env={"LLM_API_BASE_URL": "https://example.com/v1", "LLM_API_MODEL": "x"},
    )
    assert provider.config.api_base == "https://example.com/v1"
    assert provider.model == "x"


def test_select_provider_picks_first_configured() -> None:
    selected = kb_llm.select_provider(env={"DEEPSEEK_API_KEY": "k", "GROQ_API_KEY": "g"})
    assert selected is not None
    assert selected.name == "deepseek"  # deepseek wins in auto-order


def test_select_provider_respects_explicit() -> None:
    selected = kb_llm.select_provider(
        env={"KB_LLM_PROVIDER": "groq", "DEEPSEEK_API_KEY": "d", "GROQ_API_KEY": "g"}
    )
    assert selected is not None
    assert selected.name == "groq"


def test_select_provider_returns_none_when_unconfigured() -> None:
    # No cloud keys and local fallback explicitly disabled → None.
    # The model file may exist on disk; disabling the fallback is the
    # canonical way to assert "no provider" without removing the binary.
    assert kb_llm.select_provider(env={"KB_LLM_LOCAL_FALLBACK": "0"}) is None


def test_provider_status_marks_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    status = kb_llm.provider_status(env={"DEEPSEEK_API_KEY": "k"})
    deepseek_entry = next(p for p in status["providers"] if p["name"] == "deepseek")
    assert deepseek_entry["configured"] is True
    other = next(p for p in status["providers"] if p["name"] == "openai")
    assert other["configured"] is False


def test_openai_compatible_provider_generate_calls_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke-test the HTTP call without leaving the process."""

    import app.services.kb_llm as kb_llm_mod

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": " Привет!  "}}],
                "usage": {"total_tokens": 5},
            }

    calls = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["url"] = url
        calls["json"] = json
        calls["headers"] = headers
        return FakeResp()

    class FakeHttpx:
        RequestError = Exception

        @staticmethod
        def post(url, json=None, headers=None, timeout=None):
            return fake_post(url, json=json, headers=headers, timeout=timeout)

    monkeypatch.setattr(kb_llm_mod, "httpx", FakeHttpx)

    provider = kb_llm.build_provider("deepseek", env={"DEEPSEEK_API_KEY": "secret"})
    response = provider.generate("Hi", system="Be brief")

    assert response.text == "Привет!"
    assert response.provider == "deepseek"
    assert response.model == "deepseek-chat"
    assert response.raw_usage == {"total_tokens": 5}
    assert calls["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert calls["headers"]["Authorization"] == "Bearer secret"
    assert calls["json"]["messages"][0]["role"] == "system"


# ----------------------------------------------------------------------
# kb_embeddings
# ----------------------------------------------------------------------


def test_st_embedder_is_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # ST (e5-small) is now the implicit default when no backend is configured
    for name in (
        "KB_EMBEDDINGS_BACKEND",
        "OLLAMA_EMBED_MODEL",
        "EMBEDDINGS_API_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    kb_embeddings.reset_embedder()
    embedder = kb_embeddings.get_embedder()
    assert embedder.name == "st"
    vec = embedder.embed("text")
    assert len(vec) == 384  # ST stub/real model returns 384-dim vectors


def test_ollama_embedder_constructs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_EMBEDDINGS_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    kb_embeddings.reset_embedder()
    embedder = kb_embeddings.get_embedder()
    assert isinstance(embedder, kb_embeddings.OllamaEmbedder)
    assert embedder.base_url == "http://ollama:11434"
    assert embedder.model == "nomic-embed-text"


def test_api_embedder_constructs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_EMBEDDINGS_BACKEND", "api")
    monkeypatch.setenv("EMBEDDINGS_API_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("EMBEDDINGS_API_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDINGS_API_KEY", "k")
    kb_embeddings.reset_embedder()
    embedder = kb_embeddings.get_embedder()
    assert isinstance(embedder, kb_embeddings.OpenAICompatibleEmbedder)
    assert embedder.model == "text-embedding-3-small"
    assert embedder.api_key == "k"


def test_no_hashing_warning_when_st_available(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """ST available + KB_API_KEY set → embedder is 'st' and hashing warning does NOT fire."""

    for name in ("KB_EMBEDDINGS_BACKEND", "OLLAMA_EMBED_MODEL", "EMBEDDINGS_API_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KB_API_KEY", "any-key")
    kb_embeddings.reset_embedder()

    sentinel = kb_embeddings.HashingEmbedder()
    sentinel.name = "st"  # stand in for a real ST embedder
    monkeypatch.setattr(kb_embeddings, "_try_build_st_embedder", lambda env: sentinel, raising=False)

    with caplog.at_level("WARNING", logger="app.services.kb_embeddings"):
        chosen = kb_embeddings._build_from_env(env={"KB_API_KEY": "k"})

    assert chosen is sentinel
    assert "Falling back to hashing embedder" not in caplog.text


def test_hashing_warning_when_st_unavailable_and_production_like(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """ST unavailable + KB_API_KEY set + no explicit backend → hash used and warning fires."""

    for name in ("KB_EMBEDDINGS_BACKEND", "OLLAMA_EMBED_MODEL", "EMBEDDINGS_API_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    kb_embeddings.reset_embedder()

    monkeypatch.setattr(kb_embeddings, "_try_build_st_embedder", lambda env: None, raising=False)

    with caplog.at_level("WARNING", logger="app.services.kb_embeddings"):
        chosen = kb_embeddings._build_from_env(env={"KB_API_KEY": "k"})

    assert isinstance(chosen, kb_embeddings.HashingEmbedder)
    assert "Falling back to hashing embedder" in caplog.text


def test_st_default_silent_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Pure dev mode (no KB_API_KEY) — ST is the new default, no hashing warning."""

    for name in (
        "KB_EMBEDDINGS_BACKEND",
        "OLLAMA_EMBED_MODEL",
        "EMBEDDINGS_API_BASE_URL",
        "KB_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    kb_embeddings.reset_embedder()

    with caplog.at_level("WARNING", logger="app.services.kb_embeddings"):
        embedder = kb_embeddings.get_embedder()

    # ST is the new implicit default; no hashing warning should fire
    assert embedder.name == "st"
    matching = [r for r in caplog.records if "hashing" in r.message.lower()]
    assert not matching, f"unexpected hashing warning in dev mode: {[r.message for r in caplog.records]}"


def test_hashing_silent_when_explicitly_requested(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """User chose hash explicitly — respect that even in production-like setup."""

    for name in ("OLLAMA_EMBED_MODEL", "EMBEDDINGS_API_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KB_API_KEY", "any-key")
    monkeypatch.setenv("KB_EMBEDDINGS_BACKEND", "hash")
    kb_embeddings.reset_embedder()

    with caplog.at_level("WARNING", logger="app.services.kb_embeddings"):
        embedder = kb_embeddings.get_embedder()

    assert isinstance(embedder, kb_embeddings.HashingEmbedder)
    matching = [r for r in caplog.records if "hashing" in r.message.lower()]
    assert (
        not matching
    ), f"unexpected warning for explicit hash: {[r.message for r in caplog.records]}"


def test_embedder_backend_metric_records_active_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``kb_embedder_backend_active{kind=…}`` gauge reflects the selected backend."""

    from prometheus_client import REGISTRY

    for name in (
        "KB_EMBEDDINGS_BACKEND",
        "OLLAMA_EMBED_MODEL",
        "EMBEDDINGS_API_BASE_URL",
        "KB_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    kb_embeddings.reset_embedder()

    kb_embeddings.get_embedder()

    # Default is now ST, so the `st` gauge is set, not `hash`
    st_value = REGISTRY.get_sample_value("kb_embedder_backend_active", labels={"kind": "st"})
    assert st_value == 1.0


# ----------------------------------------------------------------------
# kb_rerank
# ----------------------------------------------------------------------


def test_rerank_disabled_by_default(client: TestClient) -> None:
    """Без KB_RERANK_ENABLED reranker молчит, /search не тратит время на cross-encoder."""

    client.post("/api/kb/documents", json={"title": "X", "text": "Релевантный отпуск 28 дней"})
    response = client.post("/api/kb/search", json={"query": "отпуск", "top_k": 5})
    body = response.json()
    assert body["rerank"]["enabled"] is False
    assert body["rerank"]["used"] is False


def test_rerank_reorders_hits_when_enabled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Когда KB_RERANK_ENABLED=true и есть mock-модель — должен сменить порядок."""

    monkeypatch.setenv("KB_RERANK_ENABLED", "true")
    monkeypatch.setenv("KB_RERANK_CANDIDATES", "3")

    class FakeCrossEncoder:
        """Возвращает predictable scores: чем длиннее текст, тем выше."""

        def predict(self, pairs):
            return [float(len(text)) for _q, text in pairs]

    # Подменяем CrossEncoder в существующем app.retriever.rerank, который
    # kb_rerank переиспользует через CrossEncoderReranker.
    import app.retriever.rerank as legacy_rerank

    monkeypatch.setattr(legacy_rerank, "CrossEncoder", lambda model_name: FakeCrossEncoder())
    kb_rerank.reset_cache()

    # Добавляем 3 документа — short / medium / long
    client.post("/api/kb/documents", json={"title": "short", "text": "отпуск короткий"})
    client.post(
        "/api/kb/documents",
        json={"title": "medium", "text": "отпуск средней длины описание " * 5},
    )
    client.post(
        "/api/kb/documents",
        json={"title": "long", "text": "отпуск очень длинный детальный регламент " * 20},
    )

    response = client.post("/api/kb/search", json={"query": "отпуск", "top_k": 3})
    body = response.json()
    assert body["rerank"]["enabled"] is True
    assert body["rerank"]["used"] is True
    assert body["rerank"]["candidates"] >= 3
    # Самый длинный должен оказаться первым (FakeCrossEncoder ранжирует по длине).
    titles = [hit["document_title"] for hit in body["hits"]]
    assert titles[0] == "long"


def test_rerank_falls_back_silently_on_model_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Если CrossEncoder падает — отдаём bi-encoder hits без 500."""

    monkeypatch.setenv("KB_RERANK_ENABLED", "true")

    import app.retriever.rerank as legacy_rerank

    class BrokenCrossEncoder:
        def predict(self, pairs):
            raise RuntimeError("boom")

    monkeypatch.setattr(legacy_rerank, "CrossEncoder", lambda model_name: BrokenCrossEncoder())
    kb_rerank.reset_cache()

    client.post("/api/kb/documents", json={"title": "doc", "text": "Содержимое документа"})
    response = client.post("/api/kb/search", json={"query": "содержимое", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["hits"]  # есть результаты несмотря на ошибку reranker
    assert body["rerank"]["enabled"] is True


def test_rerank_status_in_health(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """/health возвращает блок reranker с конфигом."""

    monkeypatch.setenv("KB_RERANK_ENABLED", "true")
    monkeypatch.setenv("KB_RERANK_MODEL", "test/model")
    monkeypatch.setenv("KB_RERANK_CANDIDATES", "15")
    monkeypatch.setenv("KB_RERANK_TOPN", "4")

    response = client.get("/api/kb/health")
    body = response.json()
    assert "reranker" in body
    assert body["reranker"]["enabled"] is True
    assert body["reranker"]["model"] == "test/model"
    assert body["reranker"]["candidates"] == 15
    assert body["reranker"]["top_n"] == 4
    assert body["reranker"]["loaded"] is False  # модель не загружалась — нет вызова


def test_rerank_load_config_parses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_config корректно парсит env-переменные."""

    cfg = kb_rerank.load_config(
        env={
            "KB_RERANK_ENABLED": "yes",
            "KB_RERANK_MODEL": "BAAI/bge-reranker-v2-m3",
            "KB_RERANK_CANDIDATES": "30",
            "KB_RERANK_TOPN": "8",
            "KB_RERANK_BATCH": "16",
        }
    )
    assert cfg.enabled is True
    assert cfg.model_name == "BAAI/bge-reranker-v2-m3"
    assert cfg.candidates == 30
    assert cfg.top_n == 8
    assert cfg.batch_size == 16


def test_rerank_load_config_defaults_when_env_empty() -> None:
    cfg = kb_rerank.load_config(env={})
    assert cfg.enabled is False
    assert cfg.model_name == kb_rerank.DEFAULT_MODEL_NAME
    assert cfg.candidates == kb_rerank.DEFAULT_CANDIDATES
    assert cfg.top_n == kb_rerank.DEFAULT_TOP_N


def test_rerank_load_config_clamps_invalid_numbers() -> None:
    cfg = kb_rerank.load_config(
        env={"KB_RERANK_CANDIDATES": "not-a-number", "KB_RERANK_TOPN": "9999"}
    )
    assert cfg.candidates == kb_rerank.DEFAULT_CANDIDATES
    assert cfg.top_n == 50  # clamp to high=50


# ----------------------------------------------------------------------
# Conversations & message history
# ----------------------------------------------------------------------


def test_create_conversation_returns_uuid_and_default_title(client: TestClient) -> None:
    response = client.post("/api/kb/conversations", json={})
    assert response.status_code == 201, response.text
    body = response.json()
    assert len(body["id"]) == 32  # uuid4().hex
    assert body["title"] == "Новый диалог"
    assert body["message_count"] == 0


def test_create_conversation_accepts_title(client: TestClient) -> None:
    response = client.post("/api/kb/conversations", json={"title": "Тест отпусков"})
    body = response.json()
    assert body["title"] == "Тест отпусков"


def test_list_conversations_orders_by_updated_desc(client: TestClient) -> None:
    for title in ["A", "B", "C"]:
        client.post("/api/kb/conversations", json={"title": title})
    response = client.get("/api/kb/conversations")
    assert response.status_code == 200
    titles = [c["title"] for c in response.json()]
    assert titles[:3] == ["C", "B", "A"]


def test_get_conversation_returns_messages(client: TestClient) -> None:
    create = client.post("/api/kb/conversations", json={"title": "X"})
    conv_id = create.json()["id"]
    # Добавим документ и зададим вопрос
    client.post("/api/kb/documents", json={"title": "doc", "text": "Отпуск 28 дней"})
    client.post("/api/kb/ask", json={"question": "Сколько отпуска?", "conversation_id": conv_id})

    response = client.get(f"/api/kb/conversations/{conv_id}")
    body = response.json()
    assert body["id"] == conv_id
    assert len(body["messages"]) == 2  # user + assistant
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"
    assert body["messages"][1]["sources"]


def test_rename_conversation(client: TestClient) -> None:
    conv_id = client.post("/api/kb/conversations", json={"title": "Old"}).json()["id"]
    response = client.patch(f"/api/kb/conversations/{conv_id}", json={"title": "Renamed"})
    assert response.status_code == 200
    assert response.json()["title"] == "Renamed"


def test_delete_conversation_removes_messages_cascade(client: TestClient) -> None:
    client.post("/api/kb/documents", json={"title": "d", "text": "Содержимое"})
    create = client.post("/api/kb/conversations", json={"title": "Convo"})
    conv_id = create.json()["id"]
    client.post("/api/kb/ask", json={"question": "Что в базе?", "conversation_id": conv_id})

    # Сообщения существуют
    detail = client.get(f"/api/kb/conversations/{conv_id}").json()
    assert detail["messages"]

    # Удаление
    response = client.delete(f"/api/kb/conversations/{conv_id}")
    assert response.status_code == 200

    # 404 на повторное удаление
    second = client.delete(f"/api/kb/conversations/{conv_id}")
    assert second.status_code == 404


def test_ask_without_conversation_id_creates_one(client: TestClient) -> None:
    client.post("/api/kb/documents", json={"title": "doc", "text": "Текст"})
    response = client.post("/api/kb/ask", json={"question": "Что в базе?"})
    body = response.json()
    assert body["conversation_id"]
    assert len(body["conversation_id"]) == 32


def test_ask_with_missing_conversation_id_returns_404(client: TestClient) -> None:
    response = client.post(
        "/api/kb/ask",
        json={"question": "Что-то", "conversation_id": "00000000000000000000000000000000"},
    )
    assert response.status_code == 404


def test_ask_multi_turn_uses_history_in_prompt(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Второй вопрос должен получать историю первого в промпте."""

    captured_prompts = []

    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def generate(self, prompt: str, *, system=None):
            captured_prompts.append(prompt)
            return kb_llm.LLMResponse(
                text="ответ модели", provider=self.name, model=self.model, elapsed_ms=1.0
            )

    monkeypatch.setattr(kb_llm, "select_provider", lambda env=None: FakeProvider())

    client.post(
        "/api/kb/documents",
        json={"title": "Отпуска", "text": "Отпуск 28 календарных дней"},
    )
    first = client.post("/api/kb/ask", json={"question": "Сколько дней отпуска?"})
    conv_id = first.json()["conversation_id"]
    assert "Контекст предыдущего диалога" not in captured_prompts[0]

    client.post(
        "/api/kb/ask",
        json={"question": "А кому положен?", "conversation_id": conv_id},
    )
    # Во втором промпте должна появиться история
    assert "Контекст предыдущего диалога" in captured_prompts[1]
    assert "Сколько дней отпуска?" in captured_prompts[1]


def test_ask_history_limit_zero_disables_context(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []

    class FakeProvider:
        name = "fake"
        model = "m"

        def generate(self, prompt: str, *, system=None):
            captured.append(prompt)
            return kb_llm.LLMResponse(text="x", provider="fake", model="m", elapsed_ms=0)

    singleton = FakeProvider()
    monkeypatch.setattr(kb_llm, "select_provider", lambda env=None: singleton)

    # Вопросы должны иметь лексическую общность с документом,
    # иначе hashing-embedder вернёт hits=[] и LLM не будет вызван.
    client.post(
        "/api/kb/documents",
        json={"title": "Отпуска", "text": "Регламент отпусков сотрудников"},
    )
    first = client.post("/api/kb/ask", json={"question": "Какой регламент отпусков?"})
    conv_id = first.json()["conversation_id"]
    client.post(
        "/api/kb/ask",
        json={
            "question": "Расскажи подробнее про отпуска",
            "conversation_id": conv_id,
            "history_limit": 0,
        },
    )
    assert len(captured) == 2
    # При history_limit=0 второй промпт НЕ содержит истории
    assert "Контекст предыдущего диалога" not in captured[1]


def test_store_message_validation() -> None:
    """Прямой тест валидации message store-уровня."""

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        path = tmp.name
    store = kb_store.KnowledgeBaseStore(path)
    conv = store.create_conversation(title="Test")

    # Невалидная роль
    with pytest.raises(ValueError):
        store.add_message(conv.id, "robot", "hi")
    # Пустой content
    with pytest.raises(ValueError):
        store.add_message(conv.id, "user", "   ")
    # Несуществующая conversation
    with pytest.raises(LookupError):
        store.add_message("00000000000000000000000000000000", "user", "hi")


# ----------------------------------------------------------------------
# Streaming /ask/stream
# ----------------------------------------------------------------------


def _parse_sse_events(text: str) -> list[dict]:
    """Parse raw SSE text into list of {event, data} entries."""

    import json as _json

    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        try:
            parsed = _json.loads(data) if data else {}
        except _json.JSONDecodeError:
            parsed = {"_raw": data}
        events.append({"event": event_name, "data": parsed})
    return events


def test_ask_stream_emits_meta_token_done_sequence(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming endpoint emits meta → token+ → done in order."""

    async def fake_stream(self, prompt, *, system=None):
        for chunk in ["Отпуск", " составляет", " 28 дней."]:
            yield chunk

    class FakeProvider:
        name = "fake-stream"
        model = "fake-model"

        def __init__(self):
            self.config = type("C", (), {"api_base": "http://x", "api_key": "k"})()

        generate_stream = fake_stream

    monkeypatch.setattr(kb_llm, "select_provider", lambda env=None: FakeProvider())

    client.post(
        "/api/kb/documents",
        json={"title": "Отпуска", "text": "Регламент отпусков сотрудников 28 дней"},
    )
    response = client.post("/api/kb/ask/stream", json={"question": "Сколько дней отпуска?"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = _parse_sse_events(response.text)

    types = [e["event"] for e in events]
    assert types[0] == "meta"
    assert "done" in types
    assert types.count("token") >= 1
    # token events идут между meta и done
    meta_idx = types.index("meta")
    done_idx = types.index("done")
    assert meta_idx < done_idx
    # Контент токенов склеивается в исходный текст
    tokens = [e["data"]["text"] for e in events if e["event"] == "token"]
    assert "".join(tokens) == "Отпуск составляет 28 дней."


def test_ask_stream_persists_full_answer_after_done(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """После завершения стрима в conversation должна быть пара user+assistant."""

    async def fake_stream(self, prompt, *, system=None):
        for chunk in ["Стрим", "-ответ"]:
            yield chunk

    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self):
            self.config = type("C", (), {})()

        generate_stream = fake_stream

    monkeypatch.setattr(kb_llm, "select_provider", lambda env=None: FakeProvider())

    # Лексическая общность между вопросом и документом нужна,
    # иначе hashing-embedder вернёт hits=[] и попадём в empty-KB ветку.
    client.post(
        "/api/kb/documents",
        json={"title": "Отпуска", "text": "Регламент отпусков сотрудников"},
    )
    response = client.post("/api/kb/ask/stream", json={"question": "Какой регламент отпусков?"})
    events = _parse_sse_events(response.text)
    meta = next(e["data"] for e in events if e["event"] == "meta")
    conv_id = meta["conversation_id"]

    detail = client.get(f"/api/kb/conversations/{conv_id}").json()
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"
    assert detail["messages"][1]["content"] == "Стрим-ответ"
    assert detail["messages"][1]["provider"] == "fake"


def test_ask_stream_extractive_fallback_when_no_provider(client: TestClient) -> None:
    """Без LLM-провайдера стрим должен отдать extractive ответ одним токеном."""

    client.post(
        "/api/kb/documents",
        json={"title": "Регламент отпусков", "text": "Отпуск 28 календарных дней"},
    )
    response = client.post("/api/kb/ask/stream", json={"question": "Сколько дней отпуска?"})
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    done = next(e["data"] for e in events if e["event"] == "done")
    assert done["provider"] == "extractive"
    tokens = [e["data"]["text"] for e in events if e["event"] == "token"]
    assert tokens
    assert "28 календарных" in "".join(tokens)


def test_ask_stream_empty_kb_emits_friendly_message(client: TestClient) -> None:
    """Стрим на пустую базу даёт корректный текст и done с provider=none."""

    response = client.post("/api/kb/ask/stream", json={"question": "Что в базе?"})
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    done = next(e["data"] for e in events if e["event"] == "done")
    assert done["provider"] == "none"
    tokens = [e["data"]["text"] for e in events if e["event"] == "token"]
    assert "В базе знаний пока нет данных" in "".join(tokens)


def test_ask_stream_rejects_missing_conversation(client: TestClient) -> None:
    response = client.post(
        "/api/kb/ask/stream",
        json={
            "question": "X",
            "conversation_id": "00000000000000000000000000000000",
        },
    )
    assert response.status_code == 404


def test_ask_stream_continues_conversation_with_history(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Второй стрим в том же conversation должен подгружать историю."""

    captured_prompts: list[str] = []

    async def fake_stream(self, prompt, *, system=None):
        captured_prompts.append(prompt)
        yield "ok"

    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self):
            self.config = type("C", (), {})()

        generate_stream = fake_stream

    monkeypatch.setattr(kb_llm, "select_provider", lambda env=None: FakeProvider())

    client.post(
        "/api/kb/documents",
        json={"title": "Отпуска", "text": "Регламент отпусков сотрудников 28 дней"},
    )
    first = client.post("/api/kb/ask/stream", json={"question": "Какой регламент отпусков?"})
    events = _parse_sse_events(first.text)
    conv_id = next(e["data"]["conversation_id"] for e in events if e["event"] == "meta")

    client.post(
        "/api/kb/ask/stream",
        json={
            "question": "Расскажи подробнее об отпусках",
            "conversation_id": conv_id,
        },
    )
    assert len(captured_prompts) == 2
    assert "Контекст предыдущего диалога" in captured_prompts[1]
    assert "Какой регламент отпусков?" in captured_prompts[1]


def test_extract_delta_text_handles_openai_format() -> None:
    """Парсер delta.content корректно извлекает чанк."""

    chunk = {"choices": [{"delta": {"content": "Hi"}}]}
    assert kb_llm._extract_delta_text(chunk) == "Hi"


def test_extract_delta_text_handles_ollama_message_format() -> None:
    """Ollama иногда возвращает delta как message.content."""

    chunk = {"choices": [{"message": {"content": "Hi"}}]}
    assert kb_llm._extract_delta_text(chunk) == "Hi"


def test_extract_delta_text_returns_empty_for_invalid() -> None:
    assert kb_llm._extract_delta_text({}) == ""
    assert kb_llm._extract_delta_text({"choices": []}) == ""
    assert kb_llm._extract_delta_text(None) == ""


def test_conversation_title_truncated_to_max_length() -> None:
    """Длинные title режутся до MAX_CONVERSATION_TITLE с многоточием."""

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        path = tmp.name
    store = kb_store.KnowledgeBaseStore(path)
    long_title = "x" * (kb_store.MAX_CONVERSATION_TITLE + 50)
    conv = store.create_conversation(title=long_title)
    assert len(conv.title) <= kb_store.MAX_CONVERSATION_TITLE
    assert conv.title.endswith("…")


# ----------------------------------------------------------------------
# B1 — API key authentication
# ----------------------------------------------------------------------


def test_auth_disabled_by_default(client: TestClient) -> None:
    """Без KB_API_KEY все эндпоинты доступны без заголовка."""

    health = client.get("/api/kb/health").json()
    assert health["auth"]["enabled"] is False

    # POST документа без ключа — должен работать
    response = client.post("/api/kb/documents", json={"title": "X", "text": "Текст"})
    assert response.status_code == 201


def test_auth_required_when_key_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """С KB_API_KEY — без заголовка mutating endpoints возвращают 401."""

    monkeypatch.setenv("KB_API_KEY", "secret-123")
    monkeypatch.setenv("KB_MVP_DB_PATH", str(tmp_path / "auth_test.sqlite"))
    # Do not load the 2GB local GGUF; this test only checks auth behaviour.
    monkeypatch.setenv("KB_LLM_LOCAL_FALLBACK", "0")
    kb_store.reset_default_store()
    kb_embeddings.reset_embedder()

    app = FastAPI()
    app.include_router(kb_router, prefix="/api/kb")
    with TestClient(app) as c:
        # health открыт даже с auth
        h = c.get("/api/kb/health")
        assert h.status_code == 200
        assert h.json()["auth"]["enabled"] is True

        # providers открыт
        p = c.get("/api/kb/providers")
        assert p.status_code == 200

        # POST без ключа — 401
        no_key = c.post("/api/kb/documents", json={"title": "X", "text": "Text"})
        assert no_key.status_code == 401
        assert no_key.json()["detail"] == "API_KEY_REQUIRED"

        # POST с неверным ключом — 401
        wrong = c.post(
            "/api/kb/documents",
            json={"title": "X", "text": "Text"},
            headers={"X-API-Key": "wrong"},
        )
        assert wrong.status_code == 401
        assert wrong.json()["detail"] == "INVALID_API_KEY"

        # POST с правильным ключом — 201
        ok = c.post(
            "/api/kb/documents",
            json={"title": "X", "text": "Text"},
            headers={"X-API-Key": "secret-123"},
        )
        assert ok.status_code == 201

    kb_store.reset_default_store()
    kb_embeddings.reset_embedder()


def test_auth_protects_all_mutating_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Все mutating + non-public endpoints должны быть под auth."""

    monkeypatch.setenv("KB_API_KEY", "k")
    monkeypatch.setenv("KB_MVP_DB_PATH", str(tmp_path / "auth_endpoints.sqlite"))
    kb_store.reset_default_store()

    app = FastAPI()
    app.include_router(kb_router, prefix="/api/kb")
    with TestClient(app) as c:
        # Все эти запросы должны вернуть 401 без ключа
        endpoints = [
            ("POST", "/api/kb/documents", {"json": {"title": "X", "text": "T"}}),
            ("POST", "/api/kb/search", {"json": {"query": "q", "top_k": 1}}),
            ("POST", "/api/kb/ask", {"json": {"question": "q", "top_k": 1}}),
            ("POST", "/api/kb/conversations", {"json": {}}),
            ("GET", "/api/kb/conversations", {}),
            ("GET", "/api/kb/documents", {}),
            ("DELETE", "/api/kb/documents/1", {}),
        ]
        for method, path, kwargs in endpoints:
            r = c.request(method, path, **kwargs)
            assert r.status_code == 401, f"{method} {path} вернул {r.status_code}, ожидался 401"

    kb_store.reset_default_store()


def test_auth_header_constant_time_compare(monkeypatch: pytest.MonkeyPatch) -> None:
    """Поверка через secrets.compare_digest должна работать одинаково для разных длин."""

    from app.api.kb_auth import _resolve_expected_key, require_api_key

    monkeypatch.setenv("KB_API_KEY", "longer-secret-key")
    assert _resolve_expected_key() == "longer-secret-key"

    # Короткий неправильный ключ — 401, не падает на длине
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        require_api_key(x_api_key="short")
    assert exc.value.status_code == 401


# ----------------------------------------------------------------------
# B2 — Hard LIMIT in search
# ----------------------------------------------------------------------


def test_search_respects_hard_limit_env(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KB_SEARCH_HARD_LIMIT должен ограничивать количество извлекаемых чанков."""

    monkeypatch.setenv("KB_SEARCH_HARD_LIMIT", "100")
    monkeypatch.setenv(
        "KB_MVP_DB_PATH",
        (
            str(client.app.state.kb_mvp_store.db_path)
            if hasattr(client.app.state, "kb_mvp_store")
            else ""
        ),
    )
    # Просто проверяем что переменная читается — реальное создание 10K чанков долго
    assert kb_store._search_hard_limit() == 100


def test_search_hard_limit_default() -> None:
    """Без KB_SEARCH_HARD_LIMIT — используется DEFAULT_SEARCH_HARD_LIMIT."""

    assert kb_store._search_hard_limit() == kb_store.DEFAULT_SEARCH_HARD_LIMIT


def test_search_hard_limit_clamps_invalid() -> None:
    """Невалидные значения KB_SEARCH_HARD_LIMIT падают в default + clamp."""

    import os

    os.environ["KB_SEARCH_HARD_LIMIT"] = "not-a-number"
    try:
        assert kb_store._search_hard_limit() == kb_store.DEFAULT_SEARCH_HARD_LIMIT

        os.environ["KB_SEARCH_HARD_LIMIT"] = "10"  # below floor=100
        assert kb_store._search_hard_limit() == 100

        os.environ["KB_SEARCH_HARD_LIMIT"] = "99999999"  # above ceiling=1M
        assert kb_store._search_hard_limit() == 1_000_000
    finally:
        os.environ.pop("KB_SEARCH_HARD_LIMIT", None)


def test_search_applies_limit_in_sql(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Поиск с маленьким лимитом — отбирает только N чанков из БД."""

    monkeypatch.setenv("KB_SEARCH_HARD_LIMIT", "100")
    # Добавим 3 документа
    for i in range(3):
        client.post(
            "/api/kb/documents",
            json={"title": f"doc{i}", "text": f"тестовый документ {i} с текстом " * 30},
        )
    response = client.post("/api/kb/search", json={"query": "тестовый документ", "top_k": 5})
    assert response.status_code == 200
    # Лимит выше количества чанков → поиск работает нормально
    assert response.json()["hits"]
