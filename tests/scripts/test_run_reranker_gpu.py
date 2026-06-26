"""Pure-logic tests for the turnkey reranker GPU orchestrator (no ML, no subprocess).

Covers the two testable cores: the declarative step plan (build_plan) and the
GO/NO-GO decision read from eval run-JSONs (read_metrics + decide). The subprocess
execution glue (run_step) is exercised operationally on the GPU box, not here.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import os

import pytest

import scripts.run_reranker_gpu as runner
from scripts.run_reranker_gpu import (
    build_plan,
    decide,
    is_stale,
    pending_step_names,
    preflight_problems,
    read_metrics,
)


def _names(plan) -> list[str]:
    return [s.name for s in plan]


def _by_name(plan, name):
    return next(s for s in plan if s.name == name)


# --------------------------------------------------------------------------- #
# Plan construction
# --------------------------------------------------------------------------- #
def test_full_plan_runs_seven_steps_in_canonical_order() -> None:
    plan = build_plan("full")
    assert _names(plan) == [
        "mrtydi_pairs",
        "stage1_train",
        "pravo_pairs",
        "stage2_train",
        "eval_base",
        "eval_student",
        "eval_teacher",
    ]


def test_full_stage2_chains_from_stage1_with_two_epochs() -> None:
    plan = build_plan("full")
    stage1 = _by_name(plan, "stage1_train")
    stage2 = _by_name(plan, "stage2_train")
    assert stage1.module == "scripts.train_reranker"
    assert "--epochs" in stage1.args and "2" in stage1.args
    # stage-2 starts FROM the stage-1 output dir (two-stage chaining).
    assert "--init-from" in stage2.args
    init_idx = stage2.args.index("--init-from")
    assert stage2.args[init_idx + 1] == str(stage1.produces.parent)
    assert "--lr" in stage2.args and "1e-5" in stage2.args


def test_full_train_steps_omit_device_for_cuda_autodetect() -> None:
    # On the GPU box the trainer must auto-pick cuda — never pin cpu in full mode.
    plan = build_plan("full")
    for name in ("stage1_train", "stage2_train"):
        assert "--device" not in _by_name(plan, name).args


def test_smoke_plan_uses_small_limits_one_epoch_and_cpu() -> None:
    plan = build_plan("smoke")
    mrtydi = _by_name(plan, "mrtydi_pairs")
    assert "2000" in mrtydi.args and "100000" not in mrtydi.args
    stage1 = _by_name(plan, "stage1_train")
    assert "1" in stage1.args[stage1.args.index("--epochs") + 1 : stage1.args.index("--epochs") + 2]
    assert "--device" in stage1.args and "cpu" in stage1.args


def test_eval_steps_select_model_via_rerank_env() -> None:
    plan = build_plan("full")
    base = _by_name(plan, "eval_base")
    student = _by_name(plan, "eval_student")
    teacher = _by_name(plan, "eval_teacher")
    # base = bi-encoder only, no rerank, no model env.
    assert "--rerank" not in base.args
    assert "KB_RERANK_MODEL" not in base.env
    # student reranks with the trained student dir.
    assert "--rerank" in student.args
    assert student.env["KB_RERANK_MODEL"] == str(_by_name(plan, "stage2_train").produces.parent)
    # teacher reranks with bge.
    assert "--rerank" in teacher.args
    assert teacher.env["KB_RERANK_MODEL"] == "BAAI/bge-reranker-v2-m3"


def test_every_step_declares_an_output_for_resumability() -> None:
    for step in build_plan("full"):
        assert step.produces is not None and isinstance(step.produces, Path)


def test_smoke_and_full_outputs_never_share_a_path() -> None:
    # A throwaway 1-epoch CPU smoke run must not leave artifacts that a later full
    # run would resume into — that would compute the gate verdict from the smoke
    # model. Every step's output path must differ between the two profiles.
    full = {s.name: s.produces for s in build_plan("full")}
    smoke = {s.name: s.produces for s in build_plan("smoke")}
    assert set(full) == set(smoke)
    shared = [name for name in full if full[name] == smoke[name]]
    assert shared == [], f"smoke reuses full's output path(s): {shared}"


# --------------------------------------------------------------------------- #
# Preflight (fail before burning GPU hours)
# --------------------------------------------------------------------------- #
def _plan_under(tmp_path: Path):
    """A real plan whose every ``produces`` is repointed under tmp_path, so
    pending-ness is decided by what THIS test writes, not the dev box's var/ tree.
    """
    plan = build_plan("full")
    for step in plan:
        step.produces = tmp_path / f"{step.name}.out"
    return plan


def test_pending_steps_lists_all_when_nothing_produced(tmp_path: Path) -> None:
    plan = _plan_under(tmp_path)  # nothing written yet -> every step pending
    assert pending_step_names(plan, force=False) == _names(plan)


def test_pending_steps_force_includes_every_step(tmp_path: Path) -> None:
    # --force ignores existing outputs: the full plan is always pending.
    plan = _plan_under(tmp_path)
    for step in plan:
        step.produces.write_text("done", encoding="utf-8")  # all already produced
    assert pending_step_names(plan, force=True) == _names(plan)


def test_pending_steps_skips_a_step_whose_output_exists(tmp_path: Path) -> None:
    plan = _plan_under(tmp_path)
    _by_name(plan, "mrtydi_pairs").produces.write_text("done", encoding="utf-8")
    pending = pending_step_names(plan, force=False)
    assert "mrtydi_pairs" not in pending  # produced -> skipped
    assert "stage1_train" in pending  # still missing -> pending


def test_preflight_clean_when_store_full_and_golden_present() -> None:
    problems = preflight_problems(["pravo_pairs", "eval_base"], golden_exists=True, store_docs=6141)
    assert problems == []


def test_preflight_flags_empty_store_when_pravo_pairs_pending() -> None:
    problems = preflight_problems(["pravo_pairs"], golden_exists=True, store_docs=0)
    assert len(problems) == 1
    assert "ingest_pravo" in problems[0]


def test_preflight_ignores_empty_store_when_pravo_pairs_already_done() -> None:
    # Resume: pairs already mined -> the store is no longer a prerequisite.
    problems = preflight_problems(["eval_base"], golden_exists=True, store_docs=0)
    assert problems == []


def test_preflight_flags_missing_golden_when_eval_pending() -> None:
    problems = preflight_problems(["eval_student"], golden_exists=False, store_docs=6141)
    assert len(problems) == 1
    assert "golden" in problems[0].lower()


def test_preflight_reports_both_problems_together() -> None:
    problems = preflight_problems(
        ["pravo_pairs", "eval_teacher"], golden_exists=False, store_docs=0
    )
    assert len(problems) == 2


# --------------------------------------------------------------------------- #
# Staleness: a retrained upstream invalidates its downstream eval on resume
# --------------------------------------------------------------------------- #
def _touch(path: Path, mtime: float) -> None:
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_produced_step_with_no_deps_is_not_stale(tmp_path: Path) -> None:
    plan = _plan_under(tmp_path)
    by_name = {s.name: s for s in plan}
    base = _by_name(plan, "eval_base")  # eval_base has no declared upstreams
    _touch(base.produces, 1000)
    assert is_stale(base, by_name) is False


def test_eval_student_stale_when_its_model_is_newer(tmp_path: Path) -> None:
    plan = _plan_under(tmp_path)
    by_name = {s.name: s for s in plan}
    student_eval = _by_name(plan, "eval_student")
    stage2 = _by_name(plan, "stage2_train")
    _touch(student_eval.produces, 1000)  # old eval...
    _touch(stage2.produces, 2000)  # ...but the model was retrained later
    assert is_stale(student_eval, by_name) is True
    # and the start-of-run forecast surfaces it too
    assert "eval_student" in pending_step_names(plan, force=False)


def test_eval_student_fresh_when_model_is_older(tmp_path: Path) -> None:
    plan = _plan_under(tmp_path)
    by_name = {s.name: s for s in plan}
    student_eval = _by_name(plan, "eval_student")
    stage2 = _by_name(plan, "stage2_train")
    _touch(stage2.produces, 1000)  # model trained first...
    _touch(student_eval.produces, 2000)  # ...eval ran after it -> in sync
    assert is_stale(student_eval, by_name) is False


# --------------------------------------------------------------------------- #
# Decision from eval run-JSONs
# --------------------------------------------------------------------------- #
def _write_run(path: Path, metrics: dict) -> None:
    path.write_text(_json.dumps({"surface": "mvp", "retrieval": metrics}), encoding="utf-8")


def test_read_metrics_extracts_retrieval_block(tmp_path: Path) -> None:
    p = tmp_path / "run.json"
    _write_run(p, {"hit@1": 0.8, "mrr@5": 0.9, "recall@5": 0.6})
    assert read_metrics(p) == {"hit@1": 0.8, "mrr@5": 0.9, "recall@5": 0.6}


def test_read_metrics_raises_on_empty_retrieval_block(tmp_path: Path) -> None:
    # A failed/truncated eval must blow up loudly, not return zeros that would
    # masquerade as a legitimate NO-GO verdict.
    p = tmp_path / "broken.json"
    p.write_text(_json.dumps({"surface": "mvp", "retrieval": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="no retrieval metrics"):
        read_metrics(p)


def test_decide_go_when_student_beats_base(tmp_path: Path) -> None:
    _write_run(tmp_path / "b.json", {"hit@1": 0.78, "mrr@5": 0.83, "recall@5": 0.61})
    _write_run(tmp_path / "s.json", {"hit@1": 0.85, "mrr@5": 0.92, "recall@5": 0.62})
    _write_run(tmp_path / "t.json", {"hit@1": 0.89, "mrr@5": 0.92, "recall@5": 0.66})
    out = decide(tmp_path / "b.json", tmp_path / "s.json", tmp_path / "t.json")
    assert out["verdict"] == "GO"
    assert out["passed"] is True
    assert out["teacher"]["hit@1"] == 0.89  # teacher carried for reference


def test_decide_nogo_when_student_below_base(tmp_path: Path) -> None:
    _write_run(tmp_path / "b.json", {"hit@1": 0.78, "mrr@5": 0.83, "recall@5": 0.61})
    _write_run(tmp_path / "s.json", {"hit@1": 0.06, "mrr@5": 0.15, "recall@5": 0.30})
    _write_run(tmp_path / "t.json", {"hit@1": 0.89, "mrr@5": 0.92, "recall@5": 0.66})
    out = decide(tmp_path / "b.json", tmp_path / "s.json", tmp_path / "t.json")
    assert out["verdict"] == "NO-GO"
    assert out["passed"] is False
    assert out["reasons"]


# --------------------------------------------------------------------------- #
# Resume must preserve the prior log (the runner's whole crash-tail premise)
# --------------------------------------------------------------------------- #
def test_resumed_run_appends_log_without_erasing_prior_tail(tmp_path, monkeypatch) -> None:
    # Stub out the heavy collaborators so main()'s real logging/append path runs
    # without spawning subprocesses or needing eval JSONs on disk. The plan must
    # still carry the eval steps main() reads its output paths from.
    eval_plan = [
        runner.Step("eval_base", "m", [], tmp_path / "base.json"),
        runner.Step("eval_student", "m", [], tmp_path / "student.json"),
        runner.Step("eval_teacher", "m", [], tmp_path / "teacher.json"),
    ]
    monkeypatch.setattr(runner, "build_plan", lambda profile="full": eval_plan)
    monkeypatch.setattr(runner, "run_step", lambda step, log_fh: 0)  # no subprocess
    monkeypatch.setattr(
        runner,
        "decide",
        lambda *a, **k: {
            "verdict": "GO",
            "passed": True,
            "reasons": [],
            "deltas": {},
            "base": {},
            "student": {},
            "teacher": {},
        },
    )
    log = tmp_path / "run.log"
    assert runner.main(["--skip-preflight", "--log", str(log)]) == 0
    first = log.read_text(encoding="utf-8")
    assert runner.main(["--skip-preflight", "--log", str(log)]) == 0
    second = log.read_text(encoding="utf-8")

    assert second.startswith(first)  # the first run's tail survives the resume
    assert second.count("reranker GPU run") == 2  # both run headers are present
    assert "#" * 64 in second  # runs are visually delimited
