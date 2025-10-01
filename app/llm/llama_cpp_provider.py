"""Provider built around :mod:`llama_cpp` models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from app.core.config import Settings, get_settings
from app.llm.exceptions import (
    LLMProviderError,
    LoRAAdapterNotFoundError,
    ModelNotFoundError,
    ModelNotReadyError,
)

try:  # pragma: no cover - optional dependency may be missing in tests
    from llama_cpp import Llama
except Exception:  # pragma: no cover - fall back to a stub for type checking
    class Llama:  # type: ignore[too-many-ancestors]
        def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: D401 - stub
            raise ModelNotReadyError("llama_cpp is not installed")


_GENERATION_KEYS = {"temperature", "top_p", "top_k", "max_tokens"}


@dataclass(slots=True)
class LlamaCppProvider:
    """Wraps :class:`llama_cpp.Llama` with settings-aware helpers."""

    settings: Settings = field(default_factory=get_settings)
    llama_cls: type[Llama] = Llama

    name: str = "llama-cpp"
    _model: Llama | None = field(default=None, init=False, repr=False)
    _active_adapter: Path | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    def ensure_model(self) -> None:
        """Instantiate the configured GGUF model if necessary."""

        if self._model is not None:
            return

        model_path = Path(self.settings.llm_model_path).expanduser()
        if not model_path.is_file():
            raise ModelNotFoundError(model_path)

        init_kwargs = dict(
            model_path=str(model_path),
            n_ctx=int(self.settings.llm_ctx),
            n_threads=int(self.settings.llm_threads),
            n_gpu_layers=int(self.settings.llm_gpu_layers),
        )

        try:
            self._model = self.llama_cls(**init_kwargs)
        except LLMProviderError:
            raise
        except Exception as exc:  # pragma: no cover - delegated to llama.cpp
            raise ModelNotReadyError("Failed to initialise llama.cpp model") from exc

        adapter_path = self.settings.lora_adapter_path
        if adapter_path:
            try:
                self.load_lora(adapter_path, scaling=self.settings.lora_scaling)
            except LoRAAdapterNotFoundError:
                raise
            except Exception as exc:  # pragma: no cover - llama.cpp specific
                raise ModelNotReadyError("Failed to load LoRA adapter") from exc

    # ------------------------------------------------------------------
    def _assert_ready(self) -> Llama:
        if self._model is None:
            raise ModelNotReadyError("LLM model is not initialised")
        return self._model

    # ------------------------------------------------------------------
    def generate(self, prompt: str, *, context: Mapping[str, Any] | None = None) -> str:
        """Generate a completion for *prompt* using llama.cpp."""

        self.ensure_model()
        model = self._assert_ready()

        params: dict[str, Any] = {
            "prompt": prompt,
            "temperature": float(self.settings.llm_temperature),
            "top_p": float(self.settings.llm_top_p),
            "top_k": int(self.settings.llm_top_k),
            "max_tokens": int(self.settings.llm_max_tokens),
        }

        if context:
            for key, value in context.items():
                if key in _GENERATION_KEYS:
                    params[key] = value
            options = context.get("options") if isinstance(context, Mapping) else None
            if isinstance(options, Mapping):
                for key, value in options.items():
                    if key in _GENERATION_KEYS:
                        params[key] = value

        response = model.create_completion(**params)
        choices = response.get("choices", [])
        if not choices:
            return ""
        text = choices[0].get("text", "")
        return str(text).strip()

    # ------------------------------------------------------------------
    def load_lora(self, adapter: str | Path, *, scaling: float | None = None) -> None:
        """Load a LoRA adapter and activate it for subsequent generations."""

        model = self._assert_ready()
        adapter_path = Path(adapter).expanduser()
        if not adapter_path.is_file():
            raise LoRAAdapterNotFoundError(adapter_path)

        adapter_name = adapter_path.stem
        load_adapter = getattr(model, "load_adapter", None)
        set_adapter = getattr(model, "set_adapter", None)
        if not callable(load_adapter):
            raise ModelNotReadyError("Loaded model does not support adapters")

        scale = self.settings.lora_scaling if scaling is None else scaling
        load_kwargs: dict[str, Any] = {"adapter_name": adapter_name}
        if scale is not None:
            load_kwargs["scale"] = float(scale)

        load_adapter(str(adapter_path), **load_kwargs)
        if callable(set_adapter):
            set_adapter(adapter_name=adapter_name)
        self._active_adapter = adapter_path

    # ------------------------------------------------------------------
    def unload_lora(self) -> None:
        """Unload the currently active LoRA adapter if present."""

        if self._active_adapter is None:
            return

        model = self._assert_ready()
        unload_adapter = getattr(model, "unload_adapter", None)
        if callable(unload_adapter):
            try:  # pragma: no cover - direct llama.cpp interaction
                unload_adapter(adapter_name=self._active_adapter.stem)
            finally:
                self._active_adapter = None
        else:
            self._active_adapter = None

    # ------------------------------------------------------------------
    @property
    def active_adapter(self) -> Path | None:
        """Return the path of the active LoRA adapter, if any."""

        return self._active_adapter


__all__ = ["LlamaCppProvider"]
