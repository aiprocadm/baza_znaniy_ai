"""Lightweight helpers for LoRA training artefacts.

Extracted from :mod:`scripts.train_lora` so it can be imported and tested
without pulling the heavy training stack (torch / transformers / peft).
"""

from __future__ import annotations

import shutil
from pathlib import Path


def finalise_adapter_artifacts(adapter_dir: Path) -> None:
    """Ensure both adapter filenames exist side by side in ``adapter_dir``.

    PEFT's ``save_pretrained`` writes ``adapter_model.safetensors`` (its
    standard filename). KB.AI runtime (``LORA_ADAPTER_PATH``,
    ``app/llm/lora_runtime``) references the file as
    ``adapter.safetensors``. Keep both so ``PeftModel.from_pretrained``
    on this directory still finds the PEFT-standard file while
    project-internal callers keep their stable filename.
    """
    adapter_file = adapter_dir / "adapter_model.safetensors"
    target_adapter = adapter_dir / "adapter.safetensors"
    if adapter_file.exists():
        shutil.copy2(adapter_file, target_adapter)
        return
    candidates = sorted(adapter_dir.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No safetensors adapter produced in {adapter_dir}.")
    shutil.copy2(candidates[0], target_adapter)
    if not adapter_file.exists():
        shutil.copy2(candidates[0], adapter_file)
