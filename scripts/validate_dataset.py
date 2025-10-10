#!/usr/bin/env python3
"""Validate supervised fine-tuning datasets for LoRA training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)

PROMPT_FIELDS = ("instruction", "prompt", "question")
INPUT_FIELDS = ("input", "context", "background")
RESPONSE_FIELDS = ("output", "response", "answer")


@dataclass(slots=True)
class Example:
    """Normalised dataset row."""

    instruction: str
    input: str
    output: str
    source: int


@dataclass(slots=True)
class TokenStats:
    count: int
    mean: float
    median: float
    p95: float
    maximum: int


@dataclass(slots=True)
class ValidationReport:
    dataset_size: int
    base_model: str
    max_seq_len: int
    short_threshold: int
    long_threshold: int
    prompt_stats: TokenStats
    output_stats: TokenStats
    total_stats: TokenStats
    duplicates: int
    empty_records: int
    overlong_records: int
    short_outputs: int
    long_outputs: int
    issues: list[str]


class DatasetValidationError(RuntimeError):
    """Raised when blocking issues are detected in the dataset."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LoRA training dataset")
    parser.add_argument("--path", required=True, type=Path, help="Path to JSONL dataset")
    parser.add_argument(
        "--base-model",
        required=True,
        help="Tokenizer identifier used to compute token statistics",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=int(os.getenv("LORA_TRAIN_MAX_SEQ_LEN", "4096")),
        help="Maximum sequence length expected by the training pipeline",
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=10,
        help="Outputs shorter than this number of tokens are flagged as too short",
    )
    parser.add_argument(
        "--long-threshold",
        type=int,
        default=None,
        help="Outputs longer than this number of tokens are flagged as too long",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory to write validation reports (defaults to dataset directory)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _normalise_text(record: dict[str, object], fields: Iterable[str]) -> str:
    for field in fields:
        value = record.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def load_examples(path: Path) -> list[Example]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    rows: list[Example] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:  # pragma: no cover - invalid dataset
                raise DatasetValidationError(f"Line {index}: invalid JSON ({exc})") from exc
            instruction = _normalise_text(data, PROMPT_FIELDS)
            output = _normalise_text(data, RESPONSE_FIELDS)
            context = _normalise_text(data, INPUT_FIELDS)
            if not instruction or not output:
                raise DatasetValidationError(
                    f"Line {index}: missing instruction/prompt or output/response",
                )
            rows.append(Example(instruction=instruction, input=context, output=output, source=index))
    if not rows:
        raise DatasetValidationError("Dataset is empty")
    return rows


def _token_statistics(values: list[int]) -> TokenStats:
    if not values:
        return TokenStats(count=0, mean=0.0, median=0.0, p95=0.0, maximum=0)
    if len(values) == 1:
        value = float(values[0])
        return TokenStats(count=1, mean=value, median=value, p95=value, maximum=values[0])
    try:
        p95 = float(statistics.quantiles(values, n=100, method="inclusive")[94])
    except (statistics.StatisticsError, IndexError):  # pragma: no cover - defensive guard
        p95 = float(max(values))
    return TokenStats(
        count=len(values),
        mean=float(statistics.fmean(values)),
        median=float(statistics.median(values)),
        p95=p95,
        maximum=max(values),
    )


def _format_prompt(example: Example) -> str:
    instruction = example.instruction
    context = example.input
    header = f"<s>[INST] {instruction}"
    if context:
        header += f"\n{context}"
    header += " [/INST]\n"
    return header, f"{header}{example.output}"


def validate_dataset(
    examples: Sequence[Example],
    tokenizer_name: str,
    *,
    max_seq_len: int,
    short_threshold: int,
    long_threshold: int | None,
) -> ValidationReport:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_lengths: list[int] = []
    output_lengths: list[int] = []
    total_lengths: list[int] = []
    duplicates = 0
    empty_records = 0
    overlong_records = 0
    short_outputs = 0
    long_outputs = 0
    seen_prompts: set[tuple[str, str]] = set()
    issues: list[str] = []

    for example in examples:
        prompt_key = (example.instruction, example.input)
        if prompt_key in seen_prompts:
            duplicates += 1
        else:
            seen_prompts.add(prompt_key)

        prompt_prefix, full_text = _format_prompt(example)
        prompt_tokens = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
        output_tokens = tokenizer(example.output, add_special_tokens=False)["input_ids"]
        total_tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        prompt_len = len(prompt_tokens)
        output_len = len(output_tokens)
        total_len = len(total_tokens)

        prompt_lengths.append(prompt_len)
        output_lengths.append(output_len)
        total_lengths.append(total_len)

        if prompt_len == 0 or output_len == 0:
            empty_records += 1
            issues.append(f"Line {example.source}: contains empty prompt or output")

        if total_len > max_seq_len:
            overlong_records += 1
            issues.append(
                f"Line {example.source}: token count {total_len} exceeds max_seq_len {max_seq_len}",
            )
        if output_len < short_threshold:
            short_outputs += 1
        if long_threshold is not None and output_len > long_threshold:
            long_outputs += 1

    prompt_stats = _token_statistics(prompt_lengths)
    output_stats = _token_statistics(output_lengths)
    total_stats = _token_statistics(total_lengths)

    if duplicates:
        issues.append(f"Detected {duplicates} duplicate prompts")
    if short_outputs / max(1, len(examples)) > 0.5:
        issues.append("More than 50% of outputs are very short; consider filtering")
    if long_threshold is not None and long_outputs:
        issues.append(f"{long_outputs} outputs exceed the long output threshold")

    return ValidationReport(
        dataset_size=len(examples),
        base_model=tokenizer_name,
        max_seq_len=max_seq_len,
        short_threshold=short_threshold,
        long_threshold=long_threshold or 0,
        prompt_stats=prompt_stats,
        output_stats=output_stats,
        total_stats=total_stats,
        duplicates=duplicates,
        empty_records=empty_records,
        overlong_records=overlong_records,
        short_outputs=short_outputs,
        long_outputs=long_outputs,
        issues=issues,
    )


def _report_path(dataset_path: Path, report_dir: Path, suffix: str) -> Path:
    return report_dir / f"{dataset_path.stem}_validation.{suffix}"


def write_reports(dataset_path: Path, report: ValidationReport, *, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = _report_path(dataset_path, report_dir, "json")
    md_path = _report_path(dataset_path, report_dir, "md")

    payload = asdict(report)
    payload["prompt_stats"] = asdict(report.prompt_stats)
    payload["output_stats"] = asdict(report.output_stats)
    payload["total_stats"] = asdict(report.total_stats)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines = [
        f"# Dataset validation report for `{dataset_path.name}`",
        "",
        f"* Base model: `{report.base_model}`",
        f"* Examples: {report.dataset_size}",
        f"* Max sequence length: {report.max_seq_len}",
        "",
        "## Token statistics",
        "",
        "| Segment | Mean | Median | P95 | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Prompt | {report.prompt_stats.mean:.1f} | {report.prompt_stats.median:.1f} | {report.prompt_stats.p95:.1f} | {report.prompt_stats.maximum} |",
        f"| Output | {report.output_stats.mean:.1f} | {report.output_stats.median:.1f} | {report.output_stats.p95:.1f} | {report.output_stats.maximum} |",
        f"| Total | {report.total_stats.mean:.1f} | {report.total_stats.median:.1f} | {report.total_stats.p95:.1f} | {report.total_stats.maximum} |",
        "",
        "## Quality flags",
        "",
        f"* Duplicate prompts: {report.duplicates}",
        f"* Empty records: {report.empty_records}",
        f"* Overlong records: {report.overlong_records}",
        f"* Short outputs (< {report.short_threshold} tokens): {report.short_outputs}",
    ]
    if report.long_threshold:
        md_lines.append(f"* Long outputs (> {report.long_threshold} tokens): {report.long_outputs}")
    if report.issues:
        md_lines.append("")
        md_lines.append("## Issues")
        md_lines.append("")
        for issue in report.issues:
            md_lines.append(f"* {issue}")
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = args.report_dir or args.path.parent
    long_threshold = args.long_threshold or max(args.short_threshold * 4, int(args.max_seq_len * 0.6))

    try:
        examples = load_examples(args.path)
        report = validate_dataset(
            examples,
            args.base_model,
            max_seq_len=args.max_seq_len,
            short_threshold=args.short_threshold,
            long_threshold=long_threshold,
        )
        write_reports(args.path, report, report_dir=report_dir)
    except DatasetValidationError as exc:
        LOGGER.error(str(exc))
        return 2
    except Exception:  # pragma: no cover - unexpected runtime failure
        LOGGER.exception("Dataset validation failed")
        return 3

    blocking = [issue for issue in report.issues if "exceeds" in issue or "missing" in issue]
    if report.duplicates or report.empty_records or report.overlong_records:
        blocking.append("Dataset contains blocking issues")

    if blocking:
        for item in blocking:
            LOGGER.error(item)
        return 1

    LOGGER.info("Dataset validation successful: %s", args.path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
