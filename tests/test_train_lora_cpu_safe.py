"""Tests for CPU-safe precision auto-degradation in ``scripts/train_lora.py``.

The default flags (QLoRA + fp16) are GPU-only: bitsandbytes 4-bit needs CUDA,
and HF Trainer's fp16 mixed precision needs a GPU. On a CPU-only box the run
used to crash on startup. ``resolve_cpu_safe_precision`` downgrades those flags
when no CUDA is present (with a warning) while leaving GPU behaviour untouched.

``scripts/train_lora.py`` imports torch/transformers/peft/datasets at module
load, so we inject placeholder modules (same trick as
``test_train_lora_prompt_mode.py``) to import the pure helper without the stack.
"""

from __future__ import annotations

import sys
import types


def _install_heavy_module_stubs() -> None:
    needed = ("torch", "datasets", "peft", "transformers", "transformers.trainer_callback")
    for name in needed:
        sys.modules.setdefault(name, types.ModuleType(name))
    if not hasattr(sys.modules["datasets"], "load_dataset"):
        sys.modules["datasets"].load_dataset = lambda *a, **kw: None
    for symbol in ("LoraConfig", "get_peft_model", "prepare_model_for_kbit_training"):
        if not hasattr(sys.modules["peft"], symbol):
            setattr(sys.modules["peft"], symbol, type(symbol, (), {}))
    for symbol in (
        "AutoModelForCausalLM",
        "AutoTokenizer",
        "BitsAndBytesConfig",
        "Trainer",
        "TrainingArguments",
    ):
        if not hasattr(sys.modules["transformers"], symbol):
            setattr(sys.modules["transformers"], symbol, type(symbol, (), {}))
    if not hasattr(sys.modules["transformers.trainer_callback"], "TrainerCallback"):
        sys.modules["transformers.trainer_callback"].TrainerCallback = type(
            "TrainerCallback", (), {}
        )


_install_heavy_module_stubs()


def test_cpu_disables_qlora_and_fp16_with_warnings() -> None:
    from scripts.train_lora import resolve_cpu_safe_precision

    qlora, fp16, bf16, warnings = resolve_cpu_safe_precision(
        cuda_available=False, use_qlora=True, use_fp16=True, use_bf16=False
    )
    assert (qlora, fp16) == (False, False)
    assert len(warnings) == 2  # one for QLoRA, one for fp16


def test_gpu_leaves_flags_untouched() -> None:
    from scripts.train_lora import resolve_cpu_safe_precision

    qlora, fp16, bf16, warnings = resolve_cpu_safe_precision(
        cuda_available=True, use_qlora=True, use_fp16=True, use_bf16=False
    )
    assert (qlora, fp16, bf16) == (True, True, False)
    assert warnings == []


def test_cpu_with_already_safe_config_is_noop() -> None:
    from scripts.train_lora import resolve_cpu_safe_precision

    qlora, fp16, bf16, warnings = resolve_cpu_safe_precision(
        cuda_available=False, use_qlora=False, use_fp16=False, use_bf16=True
    )
    assert (qlora, fp16, bf16) == (False, False, True)
    assert warnings == []
