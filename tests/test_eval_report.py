import json
import pathlib

import pytest

from app.eval.report import build_report, to_markdown, save_report, compare


def _retrieval():
    return {"n": 2, "aggregate": {"hit@1": 0.5, "mrr@5": 0.75}}


def test_build_and_markdown():
    rep = build_report(
        surface="mvp",
        signature={"embedder_name": "hash", "dim": 256, "doc_count": 3},
        retrieval=_retrieval(),
    )
    assert rep["surface"] == "mvp" and rep["n"] == 2
    assert rep["retrieval"]["hit@1"] == 0.5
    md = to_markdown(rep)
    assert "hash" in md and "hit@1" in md


def test_save_writes_json_and_md(tmp_path):
    rep = build_report(
        surface="mvp",
        signature={"embedder_name": "ollama", "dim": 384, "doc_count": 1},
        retrieval=_retrieval(),
    )
    out = tmp_path / "run.json"
    save_report(out, rep)
    assert json.loads(out.read_text(encoding="utf-8"))["surface"] == "mvp"
    assert (tmp_path / "run.md").exists()
    # atomic write leaves no temp behind
    assert not (tmp_path / "run.json.tmp").exists()


def test_save_report_atomic_preserves_prior_file_if_write_dies(tmp_path, monkeypatch):
    # The gate treats a report file's mere existence as a complete, parseable
    # result. A write killed mid-flight must not replace a good prior report with
    # a truncated one — the new content lives in a temp until an atomic rename, so
    # a failed promote leaves the old report intact.
    out = tmp_path / "run.json"
    out.write_text(json.dumps({"surface": "OLD"}), encoding="utf-8")
    rep = build_report(
        surface="NEW",
        signature={"embedder_name": "x", "dim": 1, "doc_count": 1},
        retrieval=_retrieval(),
    )

    def _die_on_promote(self, target):
        raise RuntimeError("killed during promote")

    monkeypatch.setattr(pathlib.Path, "replace", _die_on_promote)
    with pytest.raises(RuntimeError, match="killed during promote"):
        save_report(out, rep)
    # prior report survives — new content never reached the real path
    assert json.loads(out.read_text(encoding="utf-8"))["surface"] == "OLD"


def test_compare_emits_delta():
    a = build_report(surface="mvp", signature={}, retrieval={"aggregate": {"hit@1": 0.4}})
    b = build_report(surface="mvp", signature={}, retrieval={"aggregate": {"hit@1": 0.6}})
    out = compare(a, b)
    assert "+0.200" in out
