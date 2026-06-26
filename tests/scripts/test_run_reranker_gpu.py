"""Pure-logic tests for the turnkey reranker GPU orchestrator (no ML, no subprocess).

Covers the two testable cores: the declarative step plan (build_plan) and the
GO/NO-GO decision read from eval run-JSONs (read_metrics + decide). The subprocess
execution glue (run_step) is exercised operationally on the GPU box, not here.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

from scripts.run_reranker_gpu import build_plan, decide, read_metrics


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


# --------------------------------------------------------------------------- #
# Decision from eval run-JSONs
# --------------------------------------------------------------------------- #
def _write_run(path: Path, metrics: dict) -> None:
    path.write_text(_json.dumps({"surface": "mvp", "retrieval": metrics}), encoding="utf-8")


def test_read_metrics_extracts_retrieval_block(tmp_path: Path) -> None:
    p = tmp_path / "run.json"
    _write_run(p, {"hit@1": 0.8, "mrr@5": 0.9, "recall@5": 0.6})
    assert read_metrics(p) == {"hit@1": 0.8, "mrr@5": 0.9, "recall@5": 0.6}


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
