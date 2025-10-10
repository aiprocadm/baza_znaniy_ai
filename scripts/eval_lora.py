#!/usr/bin/env python3
"""Evaluate a LoRA adapter on a JSONL dataset using EM and ROUGE-L metrics."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import evaluate

LOGGER = logging.getLogger(__name__)
PROMPT_TEMPLATE = "<s>[INST] {instruction}\n{input} [/INST]\n"


@dataclass(slots=True)
class EvalResult:
    examples: int
    exact_match: float
    rouge_l: float
    avg_tokens: float
    no_answer_ratio: float | None
    metrics_json: Path
    report_markdown: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter")
    parser.add_argument("--base-model", required=True, help="Base model identifier or local path")
    parser.add_argument("--adapter", required=True, type=Path, help="Path to adapter directory or safetensors file")
    parser.add_argument("--dataset", required=True, type=Path, help="Evaluation dataset in JSONL format")
    parser.add_argument("--output", type=Path, default=Path("./data/lora/eval"), help="Directory for evaluation reports")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-em", type=float, default=0.0)
    parser.add_argument("--min-rouge", type=float, default=0.0)
    parser.add_argument("--no-answer-pattern", type=str, default=None, help="Regex indicating model returned no answer")
    return parser.parse_args(list(argv) if argv is not None else None)


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


def _format_prompt(instruction: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction, input=context or "")


def _normalise_text(value: str) -> str:
    cleaned = re.sub(r"[\s]+", " ", value.lower().strip())
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return cleaned


def _load_adapter(base_model: str, adapter_path: Path) -> tuple[PeftModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter_dir = adapter_path
    if adapter_path.is_file():
        adapter_dir = adapter_path.parent
    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def _generate_prediction(model: PeftModel, tokenizer: AutoTokenizer, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(1e-4, temperature) if temperature > 0 else 0.0,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    output_tokens = generated[0][input_ids.shape[-1] :]
    text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    return text.strip()


def evaluate_adapter(args: argparse.Namespace) -> EvalResult:
    adapter_path = args.adapter.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_adapter(args.base_model, adapter_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = load_dataset("json", data_files={"eval": str(args.dataset)})["eval"]

    references: list[str] = []
    predictions: list[str] = []
    no_answer = 0
    pattern = re.compile(args.no_answer_pattern, re.IGNORECASE) if args.no_answer_pattern else None
    total_tokens = 0

    for example in dataset:
        instruction, context, expected = _normalise_example(example)
        prompt = _format_prompt(instruction, context)
        generated = _generate_prediction(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        predictions.append(generated)
        references.append(expected)
        if pattern and pattern.search(generated):
            no_answer += 1
        total_tokens += len(tokenizer(generated, add_special_tokens=False)["input_ids"])

    em = _compute_exact_match(predictions, references)
    rouge_l = _compute_rouge(predictions, references)
    avg_tokens = total_tokens / max(1, len(predictions))
    no_answer_ratio = (no_answer / len(predictions)) if pattern else None

    metrics_data = {
        "exact_match": em,
        "rouge_l": rouge_l,
        "average_tokens": avg_tokens,
        "examples": len(predictions),
        "no_answer_ratio": no_answer_ratio,
    }

    json_path = output_dir / f"{args.dataset.stem}_eval.json"
    md_path = output_dir / f"{args.dataset.stem}_eval.md"
    json_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    lines = [
        f"# Evaluation report for `{args.dataset.name}`",
        "",
        f"* Examples: {len(predictions)}",
        f"* Exact match: {em:.4f}",
        f"* ROUGE-L: {rouge_l:.4f}",
        f"* Average generated tokens: {avg_tokens:.2f}",
    ]
    if no_answer_ratio is not None:
        lines.append(f"* No-answer ratio: {no_answer_ratio:.4f}")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return EvalResult(
        examples=len(predictions),
        exact_match=em,
        rouge_l=rouge_l,
        avg_tokens=avg_tokens,
        no_answer_ratio=no_answer_ratio,
        metrics_json=json_path,
        report_markdown=md_path,
    )


def _compute_exact_match(predictions: list[str], references: list[str]) -> float:
    matches = 0
    for pred, ref in zip(predictions, references):
        if _normalise_text(pred) == _normalise_text(ref):
            matches += 1
    return matches / max(1, len(predictions))


def _compute_rouge(predictions: list[str], references: list[str]) -> float:
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references, rouge_types=["rougeL"])
    return float(scores.get("rougeL", 0.0))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = evaluate_adapter(args)
    except Exception:
        LOGGER.exception("Evaluation failed")
        return 2

    success = True
    if result.exact_match < args.min_em:
        LOGGER.error("Exact match %.4f below threshold %.4f", result.exact_match, args.min_em)
        success = False
    if result.rouge_l < args.min_rouge:
        LOGGER.error("ROUGE-L %.4f below threshold %.4f", result.rouge_l, args.min_rouge)
        success = False

    print(json.dumps(asdict(result), default=str))
    return 0 if success else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
