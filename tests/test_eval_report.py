import json
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


def test_compare_emits_delta():
    a = build_report(surface="mvp", signature={}, retrieval={"aggregate": {"hit@1": 0.4}})
    b = build_report(surface="mvp", signature={}, retrieval={"aggregate": {"hit@1": 0.6}})
    out = compare(a, b)
    assert "+0.200" in out
