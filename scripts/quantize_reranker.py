"""int8 dynamic-quantization latency path for ``kbai-reranker-ru`` (v2 plan).

v1 of the distilled reranker FAILED the CPU latency gate (p50=659ms p95=1022ms
for 20 candidates; budget is p95<=200ms). The v2 runbook
(``docs/superpowers/runbooks/2026-06-10-own-reranker-training.md``) proposes
int8 dynamic quantization + shorter ``max_length`` to fit the budget. This
script builds and *measures* that path.

What it does
------------
1. Loads the trained HF reranker (``--model`` dir) through
   ``sentence_transformers.CrossEncoder`` so we score with the exact same
   tokenization / forward path as production.
2. Applies ``torch.ao.quantization.quantize_dynamic`` to the underlying
   transformer's ``nn.Linear`` layers (int8 weights, dynamic activation
   quant). This is the cleanest CPU path: no calibration set, no ONNX export
   toolchain, pure-torch and reproducible. The quantized module is scored via a
   direct HF forward (``make_direct_scorer``), NOT ``encoder.predict`` (see the
   design note below for why predict breaks on the quantized module).
3. Benchmarks p50/p95 wall time for reranking ``--candidates`` (default 20)
   per query, reusing the query groups + timing helpers from
   ``scripts.bench_reranker`` (imported, never edited), with a warm-up pass.
   Prints PASS/FAIL vs ``--budget-ms`` (default 200) for fp32 and int8.
4. Optionally serializes the quantized module with ``--save-to``.

DESIGN DECISION -- how the quantized model is served
----------------------------------------------------
``quantize_dynamic`` rewrites ``nn.Linear`` into
``torch.ao.nn.quantized.dynamic.Linear`` with *packed* int8 weights. That
state_dict is NOT loadable by ``CrossEncoder(...)`` / HF ``from_pretrained``
(those expect plain fp32 ``nn.Linear`` tensors), so the int8 artifact is
**deliberately not** a drop-in HF checkpoint.

The quantized model is therefore served as a serialized torch module written
with ``torch.save(encoder.model, ...)`` and reloaded at serve time via
``torch.load(...)``. It must be scored with a **direct HF forward**
(``tokenizer(...) -> model(**enc) -> sigmoid``; see ``make_direct_scorer``),
NOT through ``CrossEncoder.predict`` — sentence-transformers' predict dispatch
feeds the feature dict positionally into BERT and the quantized module then
trips ``input_ids.size()``. (The fp32 path tolerates this; the quantized one
does not.) Because ``torch.load`` deserializes arbitrary objects, the ``*.pt``
artifact must be treated as trusted (produced locally by this script next to
the canonical fp32 dir).

MEASURED VERDICT (2026-06-13, rubert-tiny2 student, this CPU, 20 candidates,
max_length 256): fp32 p50/p95 = 620/1228 ms, int8 = 645/852 ms — only ~1.4x on
p95 and ~0 on p50, still ~4x over the 200 ms budget. torch dynamic quant
accelerates only ``nn.Linear`` matmuls; on a tiny BERT the forward is dominated
by other ops + per-call overhead, so the win is small. **int8 alone does not
meet the gate** — the latency path needs fewer candidates, ONNX Runtime
(graph int8 + op fusion), shorter max_length, or a revised budget.

Run (heavy, needs torch + the trained model):
    py -3.13 -m scripts.quantize_reranker --compare
    py -3.13 -m scripts.quantize_reranker --save-to var/models/kbai-reranker-ru-int8.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

# Import-only reuse of the canonical latency helpers (do NOT edit bench_reranker).
from scripts.bench_reranker import group_queries, measure, percentile

DEFAULT_PAIRS = Path("var/data/rerank/pairs.jsonl")
DEFAULT_MODEL = "var/models/kbai-reranker-ru"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BUDGET_MS = 200.0


def select_queries(
    grouped: dict[str, list[str]], *, candidates: int, limit: int
) -> list[tuple[str, list[str]]]:
    """Pick queries that have at least ``candidates`` texts, capped to ``limit``.

    Pure helper (no model / torch) so it is unit-testable with a fake corpus.
    """
    qualifying = [(q, texts) for q, texts in grouped.items() if len(texts) >= candidates]
    return qualifying[:limit]


def split_warmup(
    sample: Sequence[tuple[str, list[str]]], *, warmup: int
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """Split a query sample into (warm-up, timed). Falls back to the whole
    sample for timing when it is too small to spare warm-up queries."""
    warm = list(sample[:warmup])
    timed = list(sample[warmup:]) or list(sample)
    return warm, timed


def summarize(timings: Sequence[float], *, budget_ms: float) -> tuple[float, float, bool]:
    """Return (p50, p95, passed) for a list of per-query timings (ms)."""
    p50 = percentile(timings, 0.50)
    p95 = percentile(timings, 0.95)
    return p50, p95, p95 <= budget_ms


def _load_encoder(model: str, *, max_length: int) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model, max_length=max_length)


def quantize_module(module: Any) -> Any:
    """Apply int8 dynamic quantization to all ``nn.Linear`` layers, in place.

    ``inplace=True`` swaps only the ``nn.Linear`` internals and preserves object
    identity. Note this does NOT make ``CrossEncoder.predict`` work on the result
    (predict's dict dispatch still trips the quantized module — score via
    ``make_direct_scorer`` instead); inplace is kept simply to avoid a redundant
    full copy of the model. Returns the same object for call-site convenience.
    """
    import torch
    from torch.ao.quantization import quantize_dynamic

    module.train(False)
    return quantize_dynamic(module, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)


def build_quantized_encoder(model: str, *, max_length: int) -> Any:
    """Load the fp32 CrossEncoder and swap its transformer for an int8 one.

    The tokenizer + sigmoid + ``predict`` path are untouched, so the returned
    object is API-compatible with the fp32 encoder.
    """
    encoder = _load_encoder(model, max_length=max_length)
    encoder.model = quantize_module(encoder.model)
    return encoder


def save_quantized_encoder(encoder: Any, dest: Path) -> None:
    """Serialize the (quantized) torch module -- NOT an HF checkpoint. See the
    module docstring for why this is a ``*.pt`` and how it is reloaded."""
    import torch

    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.model, dest)


def load_quantized_encoder(model: str, quantized_pt: Path, *, max_length: int) -> Any:
    """Reload a saved int8 module back into a CrossEncoder shell for serving.

    The ``*.pt`` is trusted local output of ``save_quantized_encoder``.
    """
    import torch

    encoder = _load_encoder(model, max_length=max_length)
    encoder.model = torch.load(quantized_pt, weights_only=False)
    encoder.model.train(False)
    return encoder


def make_direct_scorer(model: Any, tokenizer: Any, *, max_length: int):
    """Build a ``score_fn(pairs) -> [float]`` that calls the HF model directly.

    Bypasses ``CrossEncoder.predict`` on purpose: sentence-transformers' predict
    dispatch is incompatible with a dynamically-quantized module (it feeds the
    feature dict positionally into BERT -> ``input_ids.size()`` blows up). A
    plain ``tokenizer(...) -> model(**enc) -> sigmoid`` path works for both fp32
    and the int8 module, so it is also the apples-to-apples comparison path.
    """
    import torch

    model.eval()

    def score(pairs: Sequence[tuple[str, str]]) -> list[float]:
        enc = tokenizer(
            [q for q, _ in pairs],
            [t for _, t in pairs],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**enc).logits.reshape(-1)
        return torch.sigmoid(logits).tolist()

    return score


def _bench(
    score_fn: Any,
    sample: Sequence[tuple[str, list[str]]],
    *,
    candidates: int,
    warmup: int,
    budget_ms: float,
    label: str,
) -> bool:
    warm, timed = split_warmup(sample, warmup=warmup)
    if warm:
        measure(score_fn, warm, candidates=candidates)  # warm-up, discarded
    timings = measure(score_fn, timed, candidates=candidates)
    p50, p95, passed = summarize(timings, budget_ms=budget_ms)
    verdict = "PASS" if passed else "FAIL"
    print(
        f"[{label}] rerank {candidates} cand x {len(timings)} queries: "
        f"p50={p50:.0f}ms p95={p95:.0f}ms (budget {budget_ms:.0f}ms) -> {verdict}"
    )
    return passed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quantize_reranker")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also benchmark the fp32 encoder for an apples-to-apples speedup",
    )
    parser.add_argument(
        "--save-to",
        default=None,
        help="serialize the int8 module to this .pt path (see docstring on serving)",
    )
    args = parser.parse_args(argv)

    grouped = group_queries(Path(args.pairs))
    sample = select_queries(grouped, candidates=args.candidates, limit=args.queries)
    if not sample:
        raise SystemExit("No queries with enough candidates in the pairs file.")
    if len(sample) < 10:
        print(f"WARNING: only {len(sample)} qualifying queries -- p95 will be unreliable")

    if args.compare:
        fp32_enc = _load_encoder(args.model, max_length=args.max_length)
        fp32_score = make_direct_scorer(
            fp32_enc.model, fp32_enc.tokenizer, max_length=args.max_length
        )
        _bench(
            fp32_score,
            sample,
            candidates=args.candidates,
            warmup=args.warmup,
            budget_ms=args.budget_ms,
            label="fp32",
        )

    int8_enc = build_quantized_encoder(args.model, max_length=args.max_length)
    int8_score = make_direct_scorer(int8_enc.model, int8_enc.tokenizer, max_length=args.max_length)
    passed = _bench(
        int8_score,
        sample,
        candidates=args.candidates,
        warmup=args.warmup,
        budget_ms=args.budget_ms,
        label="int8",
    )

    if args.save_to:
        dest = Path(args.save_to)
        save_quantized_encoder(int8_enc, dest)
        print(f"saved int8 module -> {dest}")

    if not passed:
        raise SystemExit(f"FAIL: int8 p95 over budget {args.budget_ms:.0f}ms")
    print("PASS")


if __name__ == "__main__":
    main()
