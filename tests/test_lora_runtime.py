from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.llm import lora_runtime


class DummyProvider:
    def __init__(self) -> None:
        self.loaded: Path | None = None

    def ensure_model(self) -> None:  # pragma: no cover - trivial
        return None

    def load_lora(self, path: Path) -> None:
        self.loaded = Path(path)

    def unload_lora(self) -> None:
        self.loaded = None


@pytest.fixture(autouse=True)
def clear_runtime_state() -> None:
    lora_runtime._ACTIVE_ADAPTER = None  # type: ignore[attr-defined]
    from app.core import config as config_module

    config_module.get_settings.cache_clear()


@pytest.fixture()
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    registry = tmp_path / "registry"
    adapter_dir = registry / "demo"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter.gguf").write_bytes(b"adapter")
    manifest = {
        "name": "demo",
        "base": "meta-llama/Llama-3-8b-Instruct",
        "type": "gguf",
        "seq_len": 4096,
        "created_at": "2024-01-01T00:00:00Z",
    }
    (adapter_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setenv("LORA_REGISTRY_DIR", str(registry))
    monkeypatch.setenv("LLM_MODEL_NAME", manifest["base"])
    monkeypatch.setenv("LLM_MODEL_PATH", str(tmp_path / "model.gguf"))
    (tmp_path / "model.gguf").write_bytes(b"model")
    from types import SimpleNamespace

    settings_stub = SimpleNamespace(
        lora_registry_path=registry,
        llm_model_name=manifest["base"],
    )
    monkeypatch.setattr(lora_runtime, "get_settings", lambda: settings_stub)
    return registry


def test_list_adapters_returns_metadata(registry: Path) -> None:
    adapters = lora_runtime.list_adapters()
    assert len(adapters) == 1
    info = adapters[0]
    assert info.name == "demo"
    assert info.format == "gguf"
    assert info.payload.suffix == ".gguf"


def test_load_and_unload_adapter(registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = DummyProvider()
    monkeypatch.setattr(lora_runtime, "get_cached_provider", lambda settings=None: provider)

    info = lora_runtime.load_adapter("demo")
    assert provider.loaded == info.payload
    assert lora_runtime.active_adapter() is not None

    lora_runtime.unload_adapter("demo")
    assert provider.loaded is None
    assert lora_runtime.active_adapter() is None


def test_incompatible_adapter_raises(registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = DummyProvider()
    monkeypatch.setattr(lora_runtime, "get_cached_provider", lambda settings=None: provider)
    (registry / "other").mkdir()
    (registry / "other" / "manifest.json").write_text(
        json.dumps(
            {
                "name": "other",
                "base": "different-base",
                "type": "gguf",
                "seq_len": 4096,
                "created_at": "2024-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (registry / "other" / "adapter.gguf").write_bytes(b"adapter")
    with pytest.raises(lora_runtime.AdapterCompatibilityError):
        lora_runtime.load_adapter("other")
