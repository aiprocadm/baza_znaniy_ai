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
    doc = store.add_document(
        title="Doc", text="Отпуск — это оплачиваемый перерыв в работе сотрудника."
    )
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
    # Single short sentence → single chunk; chunk_key is a composite "filename:index" string
    assert ":" in hits[0].chunk_key
    # The retrieved chunk must contain the key word from the document
    assert "Отпуск" in hits[0].text or "отпуск" in hits[0].text


def test_run_refuses_hashing_without_flag(tmp_path, monkeypatch):
    import scripts.eval_rag as cli

    store, _ = _store_with_chunk(tmp_path)
    monkeypatch.setattr(cli, "get_store", lambda: store)
    golden = tmp_path / "g.jsonl"
    golden.write_text(GoldenItem("q", ("1",)).to_jsonl_line(), encoding="utf-8")
    with pytest.raises(SystemExit, match="hashing"):
        cli.cmd_run(
            cli.build_parser().parse_args(
                ["run", "--golden", str(golden), "--out", str(tmp_path / "run.json")]
            )
        )


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
    monkeypatch.setattr(
        cli,
        "_judge_provider",
        lambda: _Prov('{"faithfulness":5,"relevance":5,"completeness":5,"citation":5}'),
    )

    golden = tmp_path / "g.jsonl"
    golden.write_text(
        GoldenItem("Что такое отпуск?", ("1",), "перерыв").to_jsonl_line(), encoding="utf-8"
    )
    out = tmp_path / "run.json"
    cli.cmd_run(
        cli.build_parser().parse_args(
            ["run", "--golden", str(golden), "--out", str(out), "--allow-hashing", "--judge"]
        )
    )
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
    cli.cmd_generate(
        cli.build_parser().parse_args(
            ["generate", "--out", str(out), "--limit", "5", "--budget-usd", "100", "--yes"]
        )
    )

    from app.eval.dataset import load_golden, read_signature

    items = load_golden(out)
    assert items and items[0].question == "Что такое отпуск?"
    assert items[0].source == "auto" and items[0].relevant_chunks
    assert read_signature(out) is not None


def test_run_accepts_top_k_arg():
    import scripts.eval_rag as cli

    assert cli.build_parser().parse_args(["run", "--top-k", "8"]).top_k == 8
    assert cli.build_parser().parse_args(["run"]).top_k is None


def test_run_refuses_empty_golden(tmp_path, monkeypatch):
    import scripts.eval_rag as cli

    store, _ = _store_with_chunk(tmp_path)
    monkeypatch.setattr(cli, "get_store", lambda: store)
    golden = tmp_path / "empty.jsonl"
    golden.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="empty"):
        cli.cmd_run(
            cli.build_parser().parse_args(
                [
                    "run",
                    "--golden",
                    str(golden),
                    "--out",
                    str(tmp_path / "run.json"),
                    "--allow-hashing",
                ]
            )
        )


def test_run_warns_when_golden_has_no_signature(tmp_path, monkeypatch, capsys):
    import scripts.eval_rag as cli

    store, _ = _store_with_chunk(tmp_path)
    monkeypatch.setattr(cli, "get_store", lambda: store)
    golden = tmp_path / "g.jsonl"
    golden.write_text(GoldenItem("Что такое отпуск?", ("1",)).to_jsonl_line(), encoding="utf-8")
    cli.cmd_run(
        cli.build_parser().parse_args(
            ["run", "--golden", str(golden), "--out", str(tmp_path / "run.json"), "--allow-hashing"]
        )
    )
    assert "WARNING" in capsys.readouterr().out


def test_generate_emits_composite_keys(monkeypatch, tmp_path):
    """generate command must write composite 'filename:index' chunk keys, not raw int ids."""
    from app.services.kb_store import KnowledgeBaseStore
    import app.services.kb_store as kb_store_mod
    import scripts.eval_rag as cli
    from app.eval.adapter import build_global_id_key_map

    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"))
    store.add_document(title="Doc", text="первый абзац про оплату услуг.", filename="doc.md")
    monkeypatch.setattr(kb_store_mod, "get_store", lambda: store)
    monkeypatch.setattr(cli, "get_store", lambda: store)

    # Robustly look up the real global id of the first chunk in the store
    key_map = build_global_id_key_map(store)
    assert key_map, "store must have at least one chunk after add_document"
    first_chunk_id = next(iter(key_map))

    class _Pair:
        instruction = "Сколько стоит?"
        output = "45000"
        source_chunk_id = first_chunk_id

    class _FakeProvider:
        """Minimal provider so _gen_provider() returns something with name/model."""

        name = "fake"
        model = "fake"

    class _FakeGenerator:
        """Replaces SyntheticQAGenerator: always returns _Pair for any chunk."""

        def __init__(self, provider, **kwargs):
            pass

        def generate_for_chunk(self, *, chunks, chunk_ids, mode):
            return [_Pair()]

    monkeypatch.setattr(cli, "_gen_provider", lambda: _FakeProvider())
    monkeypatch.setattr(cli.sq, "SyntheticQAGenerator", _FakeGenerator)
    monkeypatch.setattr(cli.sq, "estimate_total_cost_usd", lambda **kw: 0.0)

    out = tmp_path / "golden_auto.jsonl"
    cli.main(["generate", "--out", str(out), "--limit", "1", "--yes"])

    from app.eval.dataset import load_golden

    items = load_golden(out)
    assert items, "generate must produce at least one golden item"
    assert all(
        ":" in k for it in items for k in it.relevant_chunks
    ), f"All relevant_chunks must be composite keys; got: {[it.relevant_chunks for it in items]}"


def test_generate_every_samples_breadth(monkeypatch, tmp_path):
    """--every N must stride-sample chunks so a --limit covers the whole corpus."""
    from app.services.kb_store import KnowledgeBaseStore
    import app.services.kb_store as kb_store_mod
    import scripts.eval_rag as cli

    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"))
    # Long text → multiple chunks in one document.
    store.add_document(title="Doc", text="абзац про оплату услуг. " * 400, filename="doc.md")
    monkeypatch.setattr(kb_store_mod, "get_store", lambda: store)
    monkeypatch.setattr(cli, "get_store", lambda: store)

    seen_chunk_ids: list[int] = []

    class _FakeProvider:
        name = "fake"
        model = "fake"

    class _FakeGenerator:
        def __init__(self, provider, **kwargs):
            pass

        def generate_for_chunk(self, *, chunks, chunk_ids, mode):
            seen_chunk_ids.extend(chunk_ids)
            return []

    monkeypatch.setattr(cli, "_gen_provider", lambda: _FakeProvider())
    monkeypatch.setattr(cli.sq, "SyntheticQAGenerator", _FakeGenerator)
    monkeypatch.setattr(cli.sq, "estimate_total_cost_usd", lambda **kw: 0.0)

    out = tmp_path / "golden_auto.jsonl"
    cli.main(["generate", "--out", str(out), "--every", "3", "--yes"])

    total = len(list(cli.sq.iter_chunks(store)))
    assert total >= 6, "test corpus must chunk into several pieces"
    assert seen_chunk_ids, "generator must have been called"
    assert len(seen_chunk_ids) == len(range(0, total, 3))
    # Stride sampling: consecutive sampled ids differ by the stride, not by 1.
    diffs = {b - a for a, b in zip(seen_chunk_ids, seen_chunk_ids[1:])}
    assert diffs == {3}


def test_generate_no_self_check_disables_consistency(monkeypatch, tmp_path):
    """--no-self-check must build the generator with check_self_consistency=False."""
    from app.services.kb_store import KnowledgeBaseStore
    import app.services.kb_store as kb_store_mod
    import scripts.eval_rag as cli

    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"))
    store.add_document(title="Doc", text="абзац про оплату услуг.", filename="doc.md")
    monkeypatch.setattr(kb_store_mod, "get_store", lambda: store)
    monkeypatch.setattr(cli, "get_store", lambda: store)

    captured: dict = {}

    class _FakeProvider:
        name = "fake"
        model = "fake"

    class _FakeGenerator:
        def __init__(self, provider, check_self_consistency=True):
            captured["check"] = check_self_consistency

        def generate_for_chunk(self, *, chunks, chunk_ids, mode):
            return []

    monkeypatch.setattr(cli, "_gen_provider", lambda: _FakeProvider())
    monkeypatch.setattr(cli.sq, "SyntheticQAGenerator", _FakeGenerator)
    monkeypatch.setattr(cli.sq, "estimate_total_cost_usd", lambda **kw: 0.0)

    out = tmp_path / "golden_auto.jsonl"
    cli.main(["generate", "--out", str(out), "--no-self-check", "--yes"])
    assert captured["check"] is False

    cli.main(["generate", "--out", str(out), "--yes"])
    assert captured["check"] is True
