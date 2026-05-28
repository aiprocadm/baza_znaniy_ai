"""Smoke tests for scripts.generate_rag_dataset CLI."""

from __future__ import annotations


def test_cli_module_imports() -> None:
    import scripts.generate_rag_dataset as cli

    assert callable(cli.parse_args)
    assert callable(cli.main)


def test_parse_args_minimal() -> None:
    from pathlib import Path

    from scripts.generate_rag_dataset import parse_args

    ns = parse_args(
        [
            "--corpus",
            "var/data/kb.sqlite",
            "--seeds",
            "var/data/seeds.jsonl",
            "--output",
            "var/data/rag.jsonl",
            "--target-pairs",
            "100",
        ]
    )
    assert ns.corpus == Path("var/data/kb.sqlite")
    assert ns.seeds == Path("var/data/seeds.jsonl")
    assert ns.output == Path("var/data/rag.jsonl")
    assert ns.target_pairs == 100
    assert ns.top_k == 3  # default


def test_cli_writes_jsonl_endtoend(tmp_path) -> None:
    """Tiny fixture: 2 seed Q&A pairs + 3 chunks in SQLite -> >=1 RAG samples.

    We let ``KnowledgeBaseStore`` create the schema and populate via
    ``add_document`` so the in-tree hashing embedder writes the
    embedding/dim/embedder columns the production search path expects.
    """
    import json

    from app.services.kb_store import KnowledgeBaseStore
    from app.services.synthetic_qa import QAPair
    from scripts.generate_rag_dataset import main

    db = tmp_path / "kb.sqlite"
    store = KnowledgeBaseStore(db_path=db)
    store.add_document("d1", "Отпуск — это перерыв в работе.")
    store.add_document("d1-part2", "Сотрудник имеет право на 28 дней отпуска.")
    store.add_document("d2", "Калибровка манометра проводится раз в год.")

    # Look up the chunk ids the store assigned so seed source_chunk_id
    # matches what the retriever will return.
    with store._connect() as conn:  # noqa: SLF001 - test reuse of internal helper
        rows = list(conn.execute("SELECT id, document_id, text FROM kb_chunks ORDER BY id"))
    assert rows, "store should have at least one chunk"
    first_chunk_id = int(rows[0][0])
    second_chunk_id = int(rows[1][0]) if len(rows) > 1 else first_chunk_id

    seeds_path = tmp_path / "seeds.jsonl"
    with seeds_path.open("w", encoding="utf-8") as fh:
        fh.write(
            QAPair(
                instruction="Что такое отпуск?",
                input="",
                output=f"Перерыв в работе. [doc_chunk:{first_chunk_id}]",
                source_chunk_id=first_chunk_id,
            ).to_jsonl_line()
        )
        fh.write(
            QAPair(
                instruction="Сколько дней отпуска?",
                input="",
                output=f"28 дней. [doc_chunk:{second_chunk_id}]",
                source_chunk_id=second_chunk_id,
            ).to_jsonl_line()
        )

    output = tmp_path / "rag.jsonl"
    # Use a larger target so the IRRELEVANT bucket gets >=1 slot — the
    # hashing embedder makes RELEVANT retrieval flaky on tiny corpora, so
    # we deliberately let the variant mix fall back to IRRELEVANT/EMPTY
    # rather than depending on retrieval quality in the smoke test.
    rc = main(
        [
            "--corpus",
            str(db),
            "--seeds",
            str(seeds_path),
            "--output",
            str(output),
            "--target-pairs",
            "10",
            "--top-k",
            "2",
        ]
    )
    assert rc == 0
    lines = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) >= 1
    assert all("retrieved_context" in line for line in lines)
    assert all("variant" in line["meta"] for line in lines)
