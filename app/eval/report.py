"""Assemble eval reports (JSON + Markdown) and diff two runs."""

from __future__ import annotations

import json
from pathlib import Path


def build_report(
    *,
    surface: str,
    signature: dict,
    retrieval: dict,
    generation: dict | None = None,
) -> dict:
    report: dict = {
        "surface": surface,
        "signature": signature,
        "n": retrieval.get("n", 0),
        "retrieval": retrieval.get("aggregate", {}),
    }
    if generation is not None:
        report["generation"] = generation.get("aggregate", {})
        report["generation_n"] = {
            "answerable": generation.get("n_answerable", 0),
            "refusal": generation.get("n_refusal", 0),
        }
    return report


def _metric_table(metrics: dict) -> list[str]:
    lines = ["| metric | value |", "|---|---|"]
    for key, val in metrics.items():
        lines.append(f"| {key} | {val:.3f} |")
    return lines


def to_markdown(report: dict) -> str:
    sig = report.get("signature", {})
    lines = [
        f"# RAG eval — surface `{report.get('surface', '?')}` (n={report.get('n', 0)})",
        "",
        f"- embedder: `{sig.get('embedder_name', '?')}` "
        f"(dim {sig.get('dim', '?')}), docs {sig.get('doc_count', '?')}",
        "",
        "## Retrieval",
        *_metric_table(report.get("retrieval", {})),
    ]
    if "generation" in report:
        lines += ["", "## Generation", *_metric_table(report["generation"])]
    return "\n".join(lines) + "\n"


def save_report(path: Path, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")


def compare(run_a: dict, run_b: dict) -> str:
    a_metrics = {**run_a.get("retrieval", {}), **run_a.get("generation", {})}
    b_metrics = {**run_b.get("retrieval", {}), **run_b.get("generation", {})}
    lines = ["# Compare", "", "| metric | A | B | Δ |", "|---|---|---|---|"]
    for key in sorted(set(a_metrics) | set(b_metrics)):
        a, b = a_metrics.get(key), b_metrics.get(key)
        if a is None or b is None:
            a_s = "—" if a is None else f"{a:.3f}"
            b_s = "—" if b is None else f"{b:.3f}"
            lines.append(f"| {key} | {a_s} | {b_s} | — |")
        else:
            lines.append(f"| {key} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    return "\n".join(lines) + "\n"
