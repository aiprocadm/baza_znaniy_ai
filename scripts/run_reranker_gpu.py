"""Turnkey runner for the Phase 1 reranker training gate (plan Task 7).

One command runs the whole two-stage pipeline end to end and self-reports a
GO/NO-GO verdict, so a rented CUDA box spends its hours on training, not on
re-deriving the step sequence and its known pitfalls:

    py -3.13 -m scripts.run_reranker_gpu             # full GPU run
    py -3.13 -m scripts.run_reranker_gpu --dry-run   # print the plan, no work
    py -3.13 -m scripts.run_reranker_gpu --profile smoke   # CPU pipeline smoke

Pipeline (each step is its OWN subprocess):
    1. build full mr-TyDi stage-1 pairs
    2. stage-1 train (general Russian ranking)
    3. mine structural pravo stage-2 pairs (bge teacher scores)
    4. stage-2 train  (--init-from stage-1; domain adaptation)
    5/6/7. three-way eval (base / student / teacher) on golden_pravo_natural
    -> student_gate verdict (spec §4): GO iff student beats base by mrr@5 OR
       hit@1 >= +min_delta AND does not regress recall@5.

Ops lessons baked in (see the headroom runbook):
- ONE torch process at a time — steps run as sequential subprocesses; this
  module itself never imports torch.
- Logs are flushed after every line (a crashed run still leaves a readable tail).
- The trainer auto-detects CUDA when ``--device`` is omitted, so the full
  profile never pins cpu; only the smoke profile (for this CPU box) does.
- Resumable: a step whose output already exists is skipped unless ``--force`` —
  but a step is re-run when an upstream output is newer (so retraining a model
  never leaves a stale eval, and thus a stale verdict, behind).

This runner is also the answer to the deferred CPU-latency item (Track B): the
production-fast reranker can only be this small distilled student, not the 568M
bge teacher — so closing this gate is what unblocks latency too.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from app.eval.pravo_gate import student_gate

# --- canonical paths (match the Phase 1 plan / runbook) --------------------- #
GOLDEN_NATURAL = Path("data/eval/golden_pravo_natural.jsonl")
MRTYDI_OUT = Path("var/data/rerank/mrtydi_pairs.jsonl")
PRAVO_OUT = Path("var/data/rerank/pravo_pairs.jsonl")
STAGE1_DIR = Path("var/models/kbai-reranker-ru-stage1")
STUDENT_DIR = Path("var/models/kbai-reranker-ru")
EVAL_DIR = Path("var/data/eval")
BASE_JSON = EVAL_DIR / "pravo_base.json"
STUDENT_JSON = EVAL_DIR / "pravo_student.json"
TEACHER_JSON = EVAL_DIR / "pravo_teacher.json"
TEACHER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_LOG = Path("var/log/reranker_gpu_run.log")
SMOKE_ROOT = Path("var/_smoke")  # throwaway CPU-smoke artifacts, isolated from full


@dataclass(frozen=True)
class ProfilePaths:
    """The output paths a profile writes to.

    ``smoke`` gets its OWN namespace (:data:`SMOKE_ROOT`) so a throwaway 1-epoch
    CPU run can never be reused — on resume — as if it were a real GPU run's model
    or eval, which would make the gate compute its GO/NO-GO verdict from the smoke
    model. ``full`` keeps the canonical plan/runbook paths.
    """

    mrtydi: Path
    pravo: Path
    stage1: Path
    student: Path
    base_json: Path
    student_json: Path
    teacher_json: Path


def profile_paths(profile: str) -> ProfilePaths:
    if profile == "smoke":
        return ProfilePaths(
            mrtydi=SMOKE_ROOT / "rerank/mrtydi_pairs.jsonl",
            pravo=SMOKE_ROOT / "rerank/pravo_pairs.jsonl",
            stage1=SMOKE_ROOT / "models/stage1",
            student=SMOKE_ROOT / "models/student",
            base_json=SMOKE_ROOT / "eval/pravo_base.json",
            student_json=SMOKE_ROOT / "eval/pravo_student.json",
            teacher_json=SMOKE_ROOT / "eval/pravo_teacher.json",
        )
    return ProfilePaths(
        mrtydi=MRTYDI_OUT,
        pravo=PRAVO_OUT,
        stage1=STAGE1_DIR,
        student=STUDENT_DIR,
        base_json=BASE_JSON,
        student_json=STUDENT_JSON,
        teacher_json=TEACHER_JSON,
    )


@dataclass
class Step:
    """One subprocess in the pipeline.

    ``produces`` is the artifact whose existence means the step is done (used for
    resume); ``env`` is merged over ``os.environ`` for that step only.
    ``depends_on`` names upstream steps: if any of their outputs is *newer* than
    this step's output, this step is stale and re-runs (make-style), so a freshly
    retrained model never leaves a stale eval behind.
    """

    name: str
    module: str
    args: list[str]
    produces: Path
    env: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)

    def argv(self) -> list[str]:
        # Same interpreter that launched the runner — no hardcoded ``py -3.13``.
        return [sys.executable, "-m", self.module, *self.args]


def build_plan(profile: str = "full") -> list[Step]:
    """Declarative step list for a profile ("full" GPU run or "smoke" CPU check).

    Output paths come from :func:`profile_paths`, so smoke and full never share a
    namespace (a smoke artifact can't be resumed into a full run — see
    :class:`ProfilePaths`).
    """
    if profile == "smoke":
        mrtydi_limit, mrtydi_negs = "2000", "8"
        stage1_epochs, stage2_epochs = "1", "1"
        train_device = ["--device", "cpu"]  # smoke runs on this CPU box
    else:
        mrtydi_limit, mrtydi_negs = "100000", "20"
        stage1_epochs, stage2_epochs = "2", "2"
        train_device = []  # full run: let the trainer auto-detect CUDA

    p = profile_paths(profile)
    golden = str(GOLDEN_NATURAL)
    return [
        Step(
            "mrtydi_pairs",
            "scripts.build_mrtydi_pairs",
            ["--limit", mrtydi_limit, "--negs", mrtydi_negs, "--out", str(p.mrtydi)],
            p.mrtydi,
        ),
        Step(
            "stage1_train",
            "scripts.train_reranker",
            [
                "--pairs",
                str(p.mrtydi),
                "--out",
                str(p.stage1),
                "--loss",
                "pairwise",
                "--epochs",
                stage1_epochs,
                *train_device,
            ],
            p.stage1 / "train_meta.json",
            depends_on=["mrtydi_pairs"],
        ),
        Step(
            "pravo_pairs",
            "scripts.build_pravo_pairs",
            ["--out", str(p.pravo)],
            p.pravo,
        ),
        Step(
            "stage2_train",
            "scripts.train_reranker",
            [
                "--pairs",
                str(p.pravo),
                "--out",
                str(p.student),
                "--init-from",
                str(p.stage1),
                "--loss",
                "pairwise",
                "--epochs",
                stage2_epochs,
                "--lr",
                "1e-5",
                *train_device,
            ],
            p.student / "train_meta.json",
            depends_on=["pravo_pairs", "stage1_train"],
        ),
        Step(
            "eval_base",
            "scripts.eval_rag",
            ["run", "--golden", golden, "--out", str(p.base_json)],
            p.base_json,
        ),
        Step(
            "eval_student",
            "scripts.eval_rag",
            ["run", "--golden", golden, "--rerank", "--out", str(p.student_json)],
            p.student_json,
            env={"KB_RERANK_MODEL": str(p.student)},
            depends_on=["stage2_train"],  # retrain -> this eval is stale -> re-runs
        ),
        Step(
            "eval_teacher",
            "scripts.eval_rag",
            ["run", "--golden", golden, "--rerank", "--out", str(p.teacher_json)],
            p.teacher_json,
            env={"KB_RERANK_MODEL": TEACHER_MODEL},
        ),
    ]


def _mtime(path: Path) -> float | None:
    """Last-modified time of ``path``, or ``None`` if it does not exist."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def is_stale(step: Step, by_name: dict[str, Step]) -> bool:
    """``True`` if ``step`` must (re)run on resume: its output is missing, or some
    declared upstream output is *newer* than it (make-style out-of-date check).

    The newer-upstream case is what stops a freshly retrained model from leaving a
    stale eval — and thus a stale GO/NO-GO verdict — in place on a no-``--force``
    resume.
    """
    out = _mtime(step.produces)
    if out is None:
        return True  # never produced
    for dep_name in step.depends_on:
        dep = by_name.get(dep_name)
        if dep is None:
            continue
        dep_mtime = _mtime(dep.produces)
        if dep_mtime is not None and dep_mtime > out:
            return True
    return False


def should_run(step: Step, by_name: dict[str, Step], *, force: bool) -> bool:
    """Single decision point shared by the preflight forecast and the executor:
    a step runs under ``--force`` or when it is stale. The executor re-asks this
    per step (not from a frozen list), so an upstream that ran earlier in the SAME
    pass correctly marks its dependents stale."""
    return force or is_stale(step, by_name)


def pending_step_names(plan: list[Step], *, force: bool) -> list[str]:
    """Names of steps that will run, as a start-of-run forecast: with ``--force``
    all of them, otherwise the stale ones (missing output or out-of-date vs an
    upstream — see :func:`is_stale`).

    Preflight checks prerequisites only for these steps — a resumed run whose
    pravo pairs already exist must not be blocked by an empty store, etc.
    """
    by_name = {s.name: s for s in plan}
    return [s.name for s in plan if should_run(s, by_name, force=force)]


def preflight_problems(
    pending: list[str],
    *,
    golden_exists: bool,
    store_docs: int,
) -> list[str]:
    """Cheap, torch-free prerequisite check run BEFORE any (multi-hour) step.

    The runner's whole point is to not waste rented GPU time; without this, a
    missing corpus or eval file only surfaces *after* stage-1 training, i.e. hours
    in. Checks only what the pending steps need: a non-empty pravo store iff
    ``pravo_pairs`` will mine, and the golden eval file iff any ``eval_*`` will run.
    Empty list = good to go.
    """
    problems: list[str] = []
    if "pravo_pairs" in pending and store_docs <= 0:
        problems.append(
            "pravo store is empty — run `py -3.13 -m scripts.ingest_pravo` "
            "first (check KB_MVP_DB_PATH)"
        )
    if any(name.startswith("eval_") for name in pending) and not golden_exists:
        problems.append(f"golden eval file missing: {GOLDEN_NATURAL} — build it before the run")
    return problems


def _count_store_documents() -> int:
    """Document count in the MVP store via a *direct* SQLite read — deliberately
    NOT ``get_store()``, which would construct an embedder (loads model weights;
    breaks the runner's "never import torch / one torch process" invariant and is
    far too heavy for a preflight). A missing DB file or missing table both mean an
    un-ingested store, i.e. zero documents.
    """
    import sqlite3

    from app.services.kb_store import _default_db_path  # env lookup only; torch-free

    db_path = Path(_default_db_path())
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        try:
            (count,) = conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()
        except sqlite3.OperationalError:
            return 0  # schema not initialised yet -> empty store
    return int(count)


def run_preflight(pending: list[str]) -> list[str]:
    """I/O wrapper: probe golden-file existence + store doc count, then delegate to
    the pure :func:`preflight_problems`. The store probe is a direct SQLite count
    (see :func:`_count_store_documents`), so the orchestrator keeps its torch-free
    invariant and the check stays sub-second. A store that cannot even be read is
    itself surfaced as a fixable prerequisite problem.
    """
    if "pravo_pairs" not in pending:
        store_docs = -1  # store not needed by any pending step; skip the probe
    else:
        try:
            store_docs = _count_store_documents()
        except Exception as exc:  # noqa: BLE001 — surface as a fixable prerequisite
            return [f"cannot read pravo store ({exc!r}) — check KB_MVP_DB_PATH"]
    return preflight_problems(pending, golden_exists=GOLDEN_NATURAL.exists(), store_docs=store_docs)


# The three metrics the gate actually consumes (student_gate / _format_verdict).
GATE_METRICS = ("hit@1", "mrr@5", "recall@5")


def read_metrics(path: Path) -> dict[str, float]:
    """Pull the aggregated retrieval metrics block from an eval_rag run JSON.

    Fail loud, not silent: a missing/empty ``retrieval`` block (truncated or
    failed eval) is raised here, not silently passed on as zeros — otherwise a
    broken eval is indistinguishable from a genuinely losing model and would
    masquerade as a NO-GO, discarding a possibly-good model after hours of GPU.
    """
    import json

    report = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = dict(report.get("retrieval", {}))
    missing = [m for m in GATE_METRICS if m not in metrics]
    if missing:
        raise ValueError(
            f"{path}: eval report has no retrieval metrics {missing} "
            f"(found keys {sorted(metrics)}) — the eval likely failed; re-run that step"
        )
    return metrics


def decide(
    base_path: Path,
    student_path: Path,
    teacher_path: Path,
    *,
    min_delta: float = 0.05,
) -> dict:
    """Three-way verdict: run the student gate vs base, carry teacher for context."""
    base = read_metrics(base_path)
    student = read_metrics(student_path)
    teacher = read_metrics(teacher_path)
    gate = student_gate(base, student, min_delta=min_delta)
    return {
        "verdict": "GO" if gate["passed"] else "NO-GO",
        "passed": gate["passed"],
        "reasons": gate["reasons"],
        "deltas": gate["deltas"],
        "base": base,
        "student": student,
        "teacher": teacher,
    }


# --------------------------------------------------------------------------- #
# Execution (operational; not unit-tested — exercised on the GPU box)
# --------------------------------------------------------------------------- #
def _log(line: str, log_fh: IO[str]) -> None:
    print(line)
    log_fh.write(line + "\n")
    log_fh.flush()  # crashed run still leaves a readable tail


def run_step(step: Step, log_fh: IO[str]) -> int:
    """Run one step as a subprocess, teeing flushed output to stdout + the log."""
    step.produces.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **step.env}
    env_note = " ".join(f"{k}={v}" for k, v in step.env.items())
    _log(
        f">>> {step.name}: {' '.join(step.argv())}" + (f"  [{env_note}]" if env_note else ""),
        log_fh,
    )
    proc = subprocess.Popen(
        step.argv(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        _log(raw.rstrip("\n"), log_fh)
    return proc.wait()


def _format_verdict(result: dict) -> str:
    lines = [
        "",
        "=" * 64,
        f"PHASE 1 GATE: {result['verdict']}",
        "=" * 64,
        f"  deltas (student - base): {result['deltas']}",
    ]
    for side in ("base", "student", "teacher"):
        m = result[side]
        lines.append(
            f"  {side:8s} hit@1={m.get('hit@1', 0):.3f} "
            f"mrr@5={m.get('mrr@5', 0):.3f} recall@5={m.get('recall@5', 0):.3f}"
        )
    for reason in result["reasons"]:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_reranker_gpu", description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--force", action="store_true", help="re-run steps even if output exists")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the prerequisite check (store/golden) before running",
    )
    parser.add_argument("--min-delta", type=float, default=0.05, help="GO threshold (spec §4)")
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    args = parser.parse_args(argv)

    plan = build_plan(args.profile)
    by_name = {s.name: s for s in plan}  # paths source of truth: dry-run, loop, decide
    pending = pending_step_names(plan, force=args.force)
    eval_outputs = (
        by_name["eval_base"].produces,
        by_name["eval_student"].produces,
        by_name["eval_teacher"].produces,
    )

    if args.dry_run:
        print(f"# plan ({args.profile}) — {len(plan)} steps; gate min_delta={args.min_delta}")
        for step in plan:
            env_note = "".join(f"{k}={v} " for k, v in step.env.items())
            mark = "RUN " if step.name in pending else "skip"
            print(f"  [{mark}] {step.name}: {env_note}{' '.join(step.argv())}  -> {step.produces}")
        problems = [] if args.skip_preflight else run_preflight(pending)
        for problem in problems:
            print(f"  preflight: {problem}")
        print(f"  verdict: student_gate({eval_outputs[0]}, {eval_outputs[1]}) vs base")
        return 0

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Append, never truncate: a resumed run must preserve the crashed run's tail
    # (the very thing the flushed log exists for). Separate runs with a rule.
    resumed = log_path.exists() and log_path.stat().st_size > 0
    with log_path.open("a", encoding="utf-8") as log_fh:
        if resumed:
            _log("\n" + "#" * 64, log_fh)
        _log(f"# reranker GPU run — profile={args.profile} min_delta={args.min_delta}", log_fh)
        if not args.skip_preflight:
            problems = run_preflight(pending)
            if problems:
                for problem in problems:
                    _log(f"!!! preflight: {problem}", log_fh)
                _log("aborting before any work — fix prerequisites and re-run", log_fh)
                return 2
        for step in plan:
            if not should_run(step, by_name, force=args.force):
                _log(f"--- skip {step.name} (up to date: {step.produces})", log_fh)
                continue
            code = run_step(step, log_fh)
            if code != 0:
                _log(f"!!! {step.name} FAILED (exit {code}) — aborting", log_fh)
                return code

        result = decide(*eval_outputs, min_delta=args.min_delta)
        _log(_format_verdict(result), log_fh)
        return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
