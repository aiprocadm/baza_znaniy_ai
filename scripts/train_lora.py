"""Utilities for training QLoRA adapters and exporting them for llama.cpp."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any
import importlib
import subprocess

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


@dataclass
class Example:
    question: str
    context: str
    answer: str


def load_examples(path: Path) -> List[Example]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise ValueError("Unsupported dataset format. Use CSV or JSONL with question/context/answer columns.")


def _load_jsonl(path: Path) -> List[Example]:
    rows: List[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            rows.append(_example_from_mapping(payload, line_no))
    return rows


def _load_csv(path: Path) -> List[Example]:
    rows: List[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file must have a header row with question, context and answer columns.")
        for line_no, row in enumerate(reader, start=2):
            rows.append(_example_from_mapping(row, line_no))
    return rows


def _example_from_mapping(mapping: Dict[str, Any], line_no: int) -> Example:
    try:
        question = str(mapping["question"]).strip()
        context = str(mapping["context"]).strip()
        answer = str(mapping["answer"]).strip()
    except KeyError as exc:
        raise ValueError(f"Missing required field {exc!s} on line {line_no}") from exc
    if not question or not answer:
        raise ValueError(f"Question and answer must be non-empty (line {line_no}).")
    return Example(question=question, context=context, answer=answer)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a QLoRA adapter and export GGML weights")
    parser.add_argument("dataset", type=Path, help="Path to CSV or JSONL dataset with question/context/answer columns")
    parser.add_argument("base_model", type=str, help="Base Hugging Face model identifier or local path")
    parser.add_argument("output_dir", type=Path, help="Directory to store training artefacts")
    parser.add_argument("--lora-r", type=int, default=16, help="Rank of the LoRA adapters (8-16 recommended)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="Alpha parameter for LoRA (16-32 recommended)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout rate for LoRA layers")
    parser.add_argument("--num-epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Maximum sequence length for tokenization")
    parser.add_argument("--logging-steps", type=int, default=10, help="Steps between logging updates")
    parser.add_argument("--save-steps", type=int, default=500, help="Steps between checkpoints")
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop training after the specified number of steps")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 when supported")
    parser.add_argument("--fp16", action="store_true", help="Use float16 when supported")
    parser.add_argument("--no-convert", action="store_true", help="Skip conversion to GGML LoRA")
    parser.add_argument("--ggml-name", default="adapter.ggml", help="Filename for the GGML LoRA output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(list(argv) if argv is not None else None)


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")
    if args.dataset.suffix.lower() not in {".csv", ".jsonl"}:
        raise ValueError("Dataset must be a CSV or JSONL file.")
    if args.lora_r <= 0:
        raise ValueError("--lora-r must be positive")
    if args.lora_alpha <= 0:
        raise ValueError("--lora-alpha must be positive")
    if not (0.0 <= args.lora_dropout < 1.0):
        raise ValueError("--lora-dropout must be in [0.0, 1.0)")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per-device-train-batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def format_example(example: Example) -> str:
    parts = [f"### Question\n{example.question.strip()}" ]
    if example.context:
        parts.append(f"### Context\n{example.context.strip()}")
    parts.append(f"### Answer\n{example.answer.strip()}")
    return "\n\n".join(parts)


@dataclass
class TrainingResult:
    adapter_dir: Path
    checkpoints_dir: Path


def train_lora_adapter(examples: List[Example], *, base_model: str, output_dir: Path, args: argparse.Namespace) -> TrainingResult:
    if not examples:
        raise ValueError("Dataset is empty after loading.")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from transformers.trainer_utils import set_seed
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("train_lora_adapter requires transformers, peft and torch packages") from exc

    LOGGER.info("Loaded %d training examples", len(examples))
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized_dataset = _build_dataset(tokenizer, examples, max_length=args.max_seq_length)

    bnb_config = _build_bnb_config()
    LOGGER.info("Loading base model %s", base_model)
    if bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=bnb_config,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=None,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_dir = output_dir / "checkpoints"
    adapter_dir = output_dir / "adapter"
    training_args = TrainingArguments(
        output_dir=str(training_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "logs"),
        save_total_limit=2,
    )

    data_collator = lambda data: _collate(tokenizer, data)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    LOGGER.info("Starting training")
    trainer.train()
    LOGGER.info("Training completed, saving adapter to %s", adapter_dir)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return TrainingResult(adapter_dir=adapter_dir, checkpoints_dir=training_dir)


def _build_bnb_config():
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        LOGGER.warning("bitsandbytes is not available, falling back to full precision training")
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )


def _build_dataset(tokenizer, examples: List[Example], max_length: int):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("datasets package is required for preprocessing") from exc

    texts = [format_example(example) for example in examples]

    def _tokenize(batch: Dict[str, List[str]]):
        result = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        result["labels"] = result["input_ids"].clone()
        return result

    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    return tokenized_dataset


def _collate(tokenizer, features: List[Dict[str, Any]]):
    import torch

    input_ids = torch.stack([feature["input_ids"] for feature in features])
    attention_mask = torch.stack([feature["attention_mask"] for feature in features])
    labels = torch.stack([feature["labels"] for feature in features])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_llama_cpp():
    return importlib.import_module("llama_cpp")


def convert_adapter_to_ggml(adapter_dir: Path, output_dir: Path, base_model: str, *, filename: str = "adapter.ggml") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    try:
        llama_cpp = load_llama_cpp()
    except ModuleNotFoundError:
        LOGGER.info("llama_cpp not available, using CLI conversion fallback")
        cmd = [
            sys.executable,
            "-m",
            "llama_cpp.convert_lora",
            "--base-model",
            base_model,
            "--adapter",
            str(adapter_dir),
            "--output",
            str(output_path),
        ]
        LOGGER.debug("Running conversion command: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return output_path

    convert_fn = None
    if hasattr(llama_cpp, "convert_lora_to_ggml"):
        convert_fn = llama_cpp.convert_lora_to_ggml
    elif hasattr(llama_cpp, "convert_lora_to_ggml_file"):
        convert_fn = llama_cpp.convert_lora_to_ggml_file

    if convert_fn is None:
        raise RuntimeError("llama_cpp does not expose a GGML conversion helper. Upgrade llama-cpp-python.")

    LOGGER.info("Converting adapter at %s to GGML format", adapter_dir)
    convert_fn(base_model=base_model, adapter_path=str(adapter_dir), output_path=str(output_path))
    return output_path


def main(argv: Iterable[str] | None = None) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    args = parse_args(argv)
    validate_args(args)
    LOGGER.info("Loading dataset from %s", args.dataset)
    examples = load_examples(args.dataset)
    training_result = train_lora_adapter(
        examples,
        base_model=args.base_model,
        output_dir=args.output_dir,
        args=args,
    )
    if args.no_convert:
        LOGGER.info("Skipping GGML conversion by request")
        return
    ggml_dir = args.output_dir / "ggml"
    convert_adapter_to_ggml(training_result.adapter_dir, ggml_dir, args.base_model, filename=args.ggml_name)
    LOGGER.info("GGML adapter saved to %s", ggml_dir / args.ggml_name)


if __name__ == "__main__":
    main()
