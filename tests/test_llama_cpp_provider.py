from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.llm.exceptions import LoRAAdapterNotFoundError, ModelNotFoundError
from app.llm.llama_cpp_provider import LlamaCppProvider


class FakeLlama:
    """Light-weight stand-in for :class:`llama_cpp.Llama`."""

    response_text = "fake-response"

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.loaded_adapters: dict[str, dict[str, object]] = {}
        self.current_adapter: str | None = None
        self.last_completion_kwargs: dict[str, object] | None = None

    def create_completion(self, **kwargs):
        self.last_completion_kwargs = kwargs
        return {"choices": [{"text": self.response_text}]}

    def load_adapter(self, path: str, *, adapter_name: str, scale: float | None = None):
        self.loaded_adapters[adapter_name] = {"path": path, "scale": scale}

    def set_adapter(self, adapter_name: str) -> None:
        self.current_adapter = adapter_name

    def unload_adapter(self, adapter_name: str) -> None:
        self.loaded_adapters.pop(adapter_name, None)
        if self.current_adapter == adapter_name:
            self.current_adapter = None


def test_ensure_model_requires_existing_file(tmp_path: Path) -> None:
    settings = Settings(
        llm_provider="llama-cpp",
        llm_model_path=str(tmp_path / "missing.gguf"),
    )
    provider = LlamaCppProvider(settings=settings, llama_cls=FakeLlama)

    with pytest.raises(ModelNotFoundError):
        provider.ensure_model()


def test_generate_uses_context_overrides(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    settings = Settings(
        llm_provider="llama-cpp",
        llm_model_path=str(model_path),
        llm_max_tokens=256,
        llm_temperature=0.5,
    )
    provider = LlamaCppProvider(settings=settings, llama_cls=FakeLlama)
    provider.ensure_model()

    FakeLlama.response_text = "ok"
    result = provider.generate("Prompt", context={"max_tokens": 42, "top_p": 0.8})

    assert result == "ok"
    llama = provider._assert_ready()
    assert llama.last_completion_kwargs is not None
    assert llama.last_completion_kwargs["prompt"] == "Prompt"
    assert llama.last_completion_kwargs["max_tokens"] == 42
    assert llama.last_completion_kwargs["top_p"] == 0.8
    assert llama.last_completion_kwargs["temperature"] == pytest.approx(0.5)


def test_lora_adapter_management(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    adapter_path = tmp_path / "adapter.safetensors"
    adapter_path.write_bytes(b"lora")
    settings = Settings(
        llm_provider="llama-cpp",
        llm_model_path=str(model_path),
        lora_scaling=0.25,
    )
    provider = LlamaCppProvider(settings=settings, llama_cls=FakeLlama)
    provider.ensure_model()

    with pytest.raises(LoRAAdapterNotFoundError):
        provider.load_lora(tmp_path / "missing.safetensors")

    provider.load_lora(adapter_path)
    llama = provider._assert_ready()
    assert provider.active_adapter == adapter_path
    assert llama.current_adapter == adapter_path.stem
    assert llama.loaded_adapters[adapter_path.stem]["scale"] == pytest.approx(0.25)

    provider.unload_lora()
    assert provider.active_adapter is None
    assert llama.current_adapter is None


def test_ensure_model_auto_loads_configured_lora(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    adapter_path = tmp_path / "auto.safetensors"
    adapter_path.write_bytes(b"lora")
    settings = Settings(
        llm_provider="llama-cpp",
        llm_model_path=str(model_path),
        lora_adapter_path=str(adapter_path),
        lora_scaling=0.75,
    )

    provider = LlamaCppProvider(settings=settings, llama_cls=FakeLlama)
    provider.ensure_model()

    llama = provider._assert_ready()
    assert provider.active_adapter == adapter_path
    assert llama.current_adapter == adapter_path.stem
    assert llama.loaded_adapters[adapter_path.stem]["scale"] == pytest.approx(0.75)

