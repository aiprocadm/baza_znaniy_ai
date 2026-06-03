"""Integration smoke tests for scripts/eval_rag.py CLI.

Uses a real KnowledgeBaseStore (SQLite, tmp_path) with HashingEmbedder so no
network or external deps are required.
"""
from __future__ import annotations

import pytest
from app.eval.adapter import EvalHit, make_mvp_retriever, compute_signature
from app.eval.dataset import GoldenItem
from app.services.kb_store import KnowledgeBaseStore
from app.services.kb_embeddings import HashingEmbedder


def _store_with_chunk(tmp_path):
    """Create a real SQLite store with one document containing a known sentence."""
    store = KnowledgeBaseStore(tmp_path / "kb.sqlite", embedder=HashingEmbedder())
    doc = store.add_document(title="Doc", text="Отпуск — это оплачиваемый перерыв в работе сотрудника.")
    return store, doc.id


def test_mvp_retriever_and_signature_on_real_store(tmp_path):
    store, _ = _store_with_chunk(tmp_path)
    sig = compute_signature(store)
    assert sig.doc_count == 1
    assert sig.embedder_name == "hash"
    assert sig.max_chunk_id >= 1
    retriever = make_mvp_retriever(store)
    hits = retriever("Что такое отпуск?", 5)
    assert hits and isinstance(hits[0], EvalHit)
    # Single short sentence → single chunk; chunk_id may be 1 or any positive int
    assert hits[0].chunk_id >= 1
    # The retrieved chunk must contain the key word from the document
    assert "Отпуск" in hits[0].text or "отпуск" in hits[0].text


def test_run_refuses_hashing_without_flag(tmp_path, monkeypatch):
    import scripts.eval_rag as cli
    store, _ = _store_with_chunk(tmp_path)
    monkeypatch.setattr(cli, "get_store", lambda: store)
    golden = tmp_path / "g.jsonl"
    golden.write_text(GoldenItem("q", (1,)).to_jsonl_line(), encoding="utf-8")
    with pytest.raises(SystemExit, match="hashing"):
        cli.cmd_run(cli.build_parser().parse_args(
            ["run", "--golden", str(golden), "--out", str(tmp_path / "run.json")]))


def test_run_includes_generation_when_judge_enabled(tmp_path, monkeypatch):
    import scripts.eval_rag as cli
    from app.eval.dataset import GoldenItem

    store, _ = _store_with_chunk(tmp_path)
    monkeypatch.setattr(cli, "get_store", lambda: store)

    class _Resp:
        def __init__(self, t):
            self.text = t

    class _Prov:
        name = model = "fake"

        def __init__(self, t):
            self._t = t

        def generate(self, prompt, *, system=None, max_tokens=None, temperature=None):
            return _Resp(self._t)

    monkeypatch.setattr(cli, "_gen_provider", lambda: _Prov("Отпуск — перерыв [1]."))
    monkeypatch.setattr(cli, "_judge_provider", lambda: _Prov(
        '{"faithfulness":5,"relevance":5,"completeness":5,"citation":5}'))

    golden = tmp_path / "g.jsonl"
    golden.write_text(GoldenItem("Что такое отпуск?", (1,), "перерыв").to_jsonl_line(), encoding="utf-8")
    out = tmp_path / "run.json"
    cli.cmd_run(cli.build_parser().parse_args(
        ["run", "--golden", str(golden), "--out", str(out), "--allow-hashing", "--judge"]))
    import json
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["generation"]["faithfulness"] == 1.0


def test_generate_builds_golden_from_corpus(tmp_path, monkeypatch):
    import scripts.eval_rag as cli

    store, _ = _store_with_chunk(tmp_path)
    monkeypatch.setattr(cli, "get_store", lambda: store)

    class _Resp:
        def __init__(self, t):
            self.text = t

    class _Teacher:
        name = "deepseek"
        model = "deepseek-chat"

        def generate(self, prompt, *, system=None, max_tokens=None, temperature=None):
            return _Resp(
                '{"instruction":"Что такое отпуск?","input":"",'
                '"output":"Отпуск — это оплачиваемый перерыв в работе сотрудника. [doc_chunk:1]"}'
            )

    monkeypatch.setattr(cli, "_gen_provider", lambda: _Teacher())

    out = tmp_path / "golden_auto.jsonl"
    cli.cmd_generate(cli.build_parser().parse_args(
        ["generate", "--out", str(out), "--limit", "5", "--budget-usd", "100", "--yes"]))

    from app.eval.dataset import load_golden, read_signature
    items = load_golden(out)
    assert items and items[0].question == "Что такое отпуск?"
    assert items[0].source == "auto" and items[0].relevant_chunk_ids
    assert read_signature(out) is not None
