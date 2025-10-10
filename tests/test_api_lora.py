from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core import config as config_module


class StubProvider:
    def __init__(self) -> None:
        self.loaded: Path | None = None

    def ensure_model(self) -> None:  # pragma: no cover - trivial stub
        return None

    def load_lora(self, path: Path) -> None:
        self.loaded = Path(path)

    def unload_lora(self) -> None:
        self.loaded = None

    def generate(self, prompt: str, *, context: dict | None = None) -> str:  # pragma: no cover - stub
        return "ok"


@pytest.fixture()
def lora_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    registry = tmp_path / "registry"
    adapter_dir = registry / "demo"
    adapter_dir.mkdir(parents=True)
    (tmp_path / "model.gguf").write_bytes(b"gguf")
    adapter_path = adapter_dir / "adapter.gguf"
    adapter_path.write_bytes(b"adapter")
    manifest = {
        "name": "demo",
        "base": "meta-llama/Llama-3-8b-Instruct",
        "type": "gguf",
        "seq_len": 4096,
        "created_at": "2024-01-01T00:00:00Z",
    }
    (adapter_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    provider = StubProvider()

    monkeypatch.setenv("LORA_REGISTRY_DIR", str(registry))
    monkeypatch.setenv("LORA_DEFAULT_ADAPTER", "none")
    monkeypatch.setenv("USE_LORA", "1")
    monkeypatch.setenv("LLM_MODEL_NAME", manifest["base"])
    monkeypatch.setenv("LLM_MODEL_PATH", str(tmp_path / "model.gguf"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path / 'db.sqlite'}")
    monkeypatch.setenv("AUTH_DISABLED_FOR_TESTS", "1")
    monkeypatch.setenv("RERANK_ENABLED", "0")

    from app.llm import cache as cache_module
    from app.llm import lora_runtime

    monkeypatch.setattr(cache_module, "get_cached_provider", lambda settings=None: provider)
    monkeypatch.setattr(lora_runtime, "get_cached_provider", lambda settings=None: provider)

    config_module.get_settings.cache_clear()

    from importlib import reload
    import app.main as app_main

    reload(app_main)
    client = TestClient(app_main.app)
    try:
        yield client
    finally:
        client.close()
        config_module.get_settings.cache_clear()


def test_list_adapters(lora_client: TestClient) -> None:
    response = lora_client.get("/admin/lora/list")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["name"] == "demo"
    assert items[0]["type"] == "gguf"


def test_load_and_unload_cycle(lora_client: TestClient) -> None:
    load = lora_client.post("/admin/lora/load", json={"name": "demo"})
    assert load.status_code == 200, load.json()
    payload = load.json()
    assert payload["loaded"] is True
    assert payload["adapter"]["name"] == "demo"

    ready = lora_client.get("/ready")
    ready_payload = ready.json()["details"]["lora"]
    assert ready_payload["status"] == "ok"
    assert ready_payload["detail"]["loaded"] is True

    unload = lora_client.post("/admin/lora/unload", json={"name": "demo"})
    assert unload.status_code == 200
    assert unload.json()["loaded"] is False


def test_load_unknown_adapter_returns_404(lora_client: TestClient) -> None:
    response = lora_client.post("/admin/lora/load", json={"name": "missing"})
    assert response.status_code == 404
