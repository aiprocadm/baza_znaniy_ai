"""Tests for the --prompt-mode flag added by W3.

``scripts/train_lora.py`` imports torch / transformers / peft / datasets
at module load. None of those are needed for the prompt-template logic
this file exercises, but the import would crash without them. We inject
empty placeholder modules into ``sys.modules`` so the import works in
test environments that lack the heavy ML stack.
"""

from __future__ import annotations

import sys
import types


def _install_heavy_module_stubs() -> None:
    """Provide minimal placeholders so ``import scripts.train_lora`` works."""

    needed: tuple[str, ...] = (
        "torch",
        "datasets",
        "peft",
        "transformers",
        "transformers.trainer_callback",
    )
    for name in needed:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        sys.modules[name] = module

    # The specific symbols train_lora imports from these packages must
    # exist as attributes — empty placeholder classes/functions suffice.
    for symbol in ("load_dataset",):
        if not hasattr(sys.modules["datasets"], symbol):
            setattr(sys.modules["datasets"], symbol, lambda *a, **kw: None)
    for symbol in (
        "LoraConfig",
        "get_peft_model",
        "prepare_model_for_kbit_training",
    ):
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


def test_prompt_template_rag_exists() -> None:
    from scripts.train_lora import PROMPT_TEMPLATE_RAG

    assert "{retrieved_context}" in PROMPT_TEMPLATE_RAG
    assert "{instruction}" in PROMPT_TEMPLATE_RAG


def test_parse_args_accepts_prompt_mode_rag() -> None:
    from scripts.train_lora import parse_args

    ns = parse_args(
        [
            "--base-model",
            "stub",
            "--train",
            "train.jsonl",
            "--output",
            "out",
            "--prompt-mode",
            "rag",
        ]
    )
    assert ns.prompt_mode == "rag"


def test_parse_args_prompt_mode_defaults_to_generic() -> None:
    from scripts.train_lora import parse_args

    ns = parse_args(
        [
            "--base-model",
            "stub",
            "--train",
            "train.jsonl",
            "--output",
            "out",
        ]
    )
    assert ns.prompt_mode == "generic"


def test_format_prompt_rag_uses_retrieved_context() -> None:
    from scripts.train_lora import format_prompt

    out = format_prompt(
        instruction="Что такое отпуск?",
        context="",
        retrieved_context="Фрагмент [doc_chunk:7]: Отпуск — перерыв.",
        prompt_mode="rag",
    )
    assert "Фрагмент [doc_chunk:7]" in out
    assert "Что такое отпуск?" in out


def test_format_prompt_generic_ignores_retrieved_context() -> None:
    """Backwards compat: generic mode behaves like before W3."""
    from scripts.train_lora import format_prompt

    out = format_prompt(
        instruction="Hi",
        context="extra",
        retrieved_context="should be ignored",
        prompt_mode="generic",
    )
    assert "should be ignored" not in out
    assert "Hi" in out
