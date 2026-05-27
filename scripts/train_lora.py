#!/usr/bin/env python3
"""Train a PEFT LoRA adapter with optional QLoRA quantisation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

LOGGER = logging.getLogger(__name__)
PROMPT_TEMPLATE = "<s>[INST] {instruction}\n{input} [/INST]\n"


@dataclass(slots=True)
class TrainingConfig:
    base_model: str
    train_path: Path
    eval_path: Path | None
    output_dir: Path
    max_seq_len: int
    epochs: float
    learning_rate: float
    batch_size: int
    grad_accumulation: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str] | None
    use_qlora: bool
    use_fp16: bool
    use_bf16: bool
    seed: int
    logging_steps: int


class JsonLogCallback(TrainerCallback):
    """Write training metrics as JSON lines for easy ingestion."""

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._handle = None

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("w", encoding="utf-8")
        record = {"step": state.global_step, **logs}
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    env = os.getenv
    parser = argparse.ArgumentParser(description="Train a LoRA adapter")
    parser.add_argument("--base-model", required=True, help="Base model identifier or local path")
    parser.add_argument(
        "--train", required=True, type=Path, help="Training dataset in JSONL format"
    )
    parser.add_argument("--eval", type=Path, default=None, help="Optional evaluation dataset")
    parser.add_argument(
        "--output", required=True, type=Path, help="Directory to store training artefacts"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=int(env("LORA_TRAIN_MAX_SEQ_LEN", "4096"))
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="",
        help="Comma separated list of target modules (defaults to auto-detect)",
    )
    parser.add_argument(
        "--use-qlora",
        dest="use_qlora",
        action="store_true",
        default=_env_flag("LORA_USE_QLORA", True),
    )
    parser.add_argument("--no-qlora", dest="use_qlora", action="store_false")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--fp16", action="store_true", default=_env_flag("LORA_FP16", True))
    precision.add_argument("--bf16", action="store_true", default=_env_flag("LORA_BF16", False))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=25)
    return parser.parse_args(list(argv) if argv is not None else None)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_config(args: argparse.Namespace) -> TrainingConfig:
    target_modules = [
        item.strip() for item in args.target_modules.split(",") if item.strip()
    ] or None
    if args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("LoRA dropout must be within [0, 1)")
    if args.max_seq_len <= 0:
        raise ValueError("Maximum sequence length must be positive")
    return TrainingConfig(
        base_model=args.base_model,
        train_path=args.train,
        eval_path=args.eval,
        output_dir=args.output,
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accumulation=args.gradient_accumulation,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        use_qlora=bool(args.use_qlora),
        use_fp16=bool(args.fp16),
        use_bf16=bool(args.bf16),
        seed=args.seed,
        logging_steps=max(1, int(args.logging_steps)),
    )


def _format_prompt(instruction: str, context: str) -> str:
    context_block = context or ""
    return PROMPT_TEMPLATE.format(instruction=instruction, input=context_block)


def _normalise_example(example: dict[str, Any]) -> tuple[str, str, str]:
    def pick(fields: tuple[str, ...]) -> str:
        for field in fields:
            value = example.get(field)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    instruction = pick(("instruction", "prompt", "question"))
    output = pick(("output", "response", "answer"))
    context = pick(("input", "context", "background"))
    if not instruction or not output:
        raise ValueError("Dataset rows must contain instruction/prompt and output/response fields")
    return instruction, context, output


def _build_feature(
    example: dict[str, Any], tokenizer: AutoTokenizer, *, max_seq_len: int
) -> dict[str, list[int]]:
    instruction, context, output = _normalise_example(example)
    prompt_prefix = _format_prompt(instruction, context)
    prompt_tokens = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
    response_tokens = tokenizer(output, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        response_tokens = response_tokens + [tokenizer.eos_token_id]

    combined = prompt_tokens + response_tokens
    if len(combined) > max_seq_len:
        overflow = len(combined) - max_seq_len
        if overflow >= len(prompt_tokens):
            overflow -= len(prompt_tokens)
            prompt_tokens = []
            if overflow > 0:
                response_tokens = response_tokens[overflow:]
        else:
            prompt_tokens = prompt_tokens[overflow:]
        combined = prompt_tokens + response_tokens
        if len(combined) > max_seq_len:
            combined = combined[-max_seq_len:]

    labels_core = [-100] * len(prompt_tokens) + response_tokens
    if len(labels_core) > len(combined):
        labels_core = labels_core[-len(combined) :]
    elif len(labels_core) < len(combined):
        labels_core = [-100] * (len(combined) - len(labels_core)) + labels_core

    pad_id = tokenizer.pad_token_id
    padding = max_seq_len - len(combined)
    input_ids = [pad_id] * padding + combined
    labels = [-100] * padding + labels_core
    attention_mask = [0] * padding + [1] * len(combined)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _tokenizer_for_base(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_datasets(config: TrainingConfig, tokenizer: AutoTokenizer):
    data_files: dict[str, str] = {"train": str(config.train_path)}
    if config.eval_path:
        data_files["validation"] = str(config.eval_path)

    raw = load_dataset("json", data_files=data_files)

    def _process(example: dict[str, Any]) -> dict[str, Any]:
        return _build_feature(example, tokenizer, max_seq_len=config.max_seq_len)

    train_dataset = raw["train"].map(_process, remove_columns=raw["train"].column_names)
    eval_dataset = None
    if "validation" in raw:
        eval_dataset = raw["validation"].map(
            _process, remove_columns=raw["validation"].column_names
        )

    train_dataset.set_format(type="torch")
    if eval_dataset is not None:
        eval_dataset.set_format(type="torch")

    return train_dataset, eval_dataset


def _build_model(config: TrainingConfig, tokenizer: AutoTokenizer) -> torch.nn.Module:
    dtype = torch.float32
    if config.use_bf16:
        dtype = torch.bfloat16
    elif config.use_fp16:
        dtype = torch.float16

    if config.use_qlora:
        compute_dtype = dtype if dtype != torch.float32 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(base_model)
        model.gradient_checkpointing_enable()
    else:
        model = AutoModelForCausalLM.from_pretrained(config.base_model)
        if config.use_fp16 or config.use_bf16:
            model = model.to(dtype)

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def _training_arguments(
    config: TrainingConfig, run_dir: Path, *, evaluation: bool
) -> TrainingArguments:
    logging_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoints"
    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        overwrite_output_dir=True,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accumulation,
        warmup_steps=0,
        logging_steps=config.logging_steps,
        logging_dir=str(logging_dir),
        save_total_limit=2,
        evaluation_strategy="epoch" if evaluation else "no",
        bf16=config.use_bf16,
        fp16=config.use_fp16 and not config.use_bf16,
        gradient_checkpointing=config.use_qlora,
        report_to=["tensorboard"],
        seed=config.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = _load_config(args)
    except Exception as exc:
        LOGGER.error("Invalid configuration: %s", exc)
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_short = (
        Path(config.base_model).name
        if Path(config.base_model).exists()
        else config.base_model.split("/")[-1]
    )
    run_name = f"{timestamp}_{base_short}_r{config.lora_r}_a{config.lora_alpha}"
    run_dir = config.output_dir / run_name
    adapter_dir = run_dir / "adapter"
    logs_path = run_dir / "logs" / "training.jsonl"

    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _tokenizer_for_base(config.base_model)
    train_dataset, eval_dataset = _load_datasets(config, tokenizer)

    model = _build_model(config, tokenizer)

    training_args = _training_arguments(config, run_dir, evaluation=eval_dataset is not None)
    json_logger = JsonLogCallback(logs_path)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[json_logger],
    )

    torch.manual_seed(config.seed)

    train_result = trainer.train()
    trainer.save_state()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)

    adapter_file = adapter_dir / "adapter_model.safetensors"
    target_adapter = adapter_dir / "adapter.safetensors"
    if adapter_file.exists():
        adapter_file.replace(target_adapter)
    else:
        # Fallback to the first safetensors file created by `save_pretrained`.
        candidates = sorted(adapter_dir.glob("*.safetensors"))
        if candidates:
            candidates[0].replace(target_adapter)
        else:
            raise FileNotFoundError(f"No safetensors adapter produced in {adapter_dir}.")
    tokenizer.save_pretrained(run_dir / "tokenizer")

    metrics = train_result.metrics
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    state_path = run_dir / "trainer_state.json"
    checkpoint_state = Path(training_args.output_dir) / "trainer_state.json"
    if checkpoint_state.exists():
        state_path.write_text(checkpoint_state.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "run_name": run_name,
        "adapter_name": run_name,
        "adapter_dir": str(adapter_dir),
        "adapter_path": str(target_adapter),
        "metrics_path": str(metrics_path),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
