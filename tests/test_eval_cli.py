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
