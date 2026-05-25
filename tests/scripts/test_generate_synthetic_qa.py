"""CLI smoke tests for scripts/generate_synthetic_qa.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.kb_llm import LLMResponse
from app.services.kb_store import KnowledgeBaseStore


class _FakeProvider:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return "deepseek-chat"  # so pricing table lookup succeeds

    def generate(self, prompt, *, system=None, max_tokens=None, temperature=None):
        self.calls += 1
        text = self._responses[(self.calls - 1) % len(self._responses)]
        return LLMResponse(text=text, provider="deepseek", model="deepseek-chat", elapsed_ms=1.0)


@pytest.fixture()
def populated_store(tmp_path: Path) -> KnowledgeBaseStore:
    store = KnowledgeBaseStore(db_path=tmp_path / "kb.sqlite")
    store.add_document(
        title="Reg",
        text="The annual leave is twenty-eight days. " * 30,
    )
    return store


def test_cli_writes_jsonl_consumable_by_validate_dataset(tmp_path, populated_store, monkeypatch):
    import sys
    import types

    # ``scripts.validate_dataset`` imports ``transformers`` at module load time
    # for its tokenizer-based validation step.  ``load_examples`` itself does
    # not touch the tokenizer, so we install a lightweight stub when the real
    # package is unavailable to keep the test hermetic.
    if "transformers" not in sys.modules:
        try:  # pragma: no cover - exercised only when real package is present
            import transformers  # noqa: F401
        except ModuleNotFoundError:
            stub = types.ModuleType("transformers")
            stub.AutoTokenizer = type("AutoTokenizer", (), {})
            sys.modules["transformers"] = stub

    from scripts import generate_synthetic_qa as cli
    from scripts import validate_dataset as vd

    response = (
        '{"instruction":"What is the annual leave?","input":"",'
        '"output":"The annual leave is twenty-eight calendar days per employee per year. [doc_chunk:1]"}'
    )
    fake_provider = _FakeProvider(responses=[response])

    monkeypatch.setattr(cli, "_load_store", lambda args: populated_store)
    monkeypatch.setattr(cli, "_load_provider", lambda args: fake_provider)

    out_path = tmp_path / "out.jsonl"
    exit_code = cli.main(
        [
            "--corpus", str(tmp_path / "kb.sqlite"),
            "--provider", "deepseek",
            "--mode", "single",
            "--output", str(out_path),
            "--no-self-consistency",
            "--no-budget-guard",
        ]
    )

    assert exit_code == 0
    assert out_path.exists()

    examples = vd.load_examples(out_path)
    assert len(examples) >= 1


def test_cli_resume_skips_processed_chunks(tmp_path, populated_store, monkeypatch):
    from scripts import generate_synthetic_qa as cli

    response = (
        '{"instruction":"What is the annual leave?","input":"",'
        '"output":"The annual leave is twenty-eight calendar days per employee per year. [doc_chunk:1]"}'
    )
    fake_provider = _FakeProvider(responses=[response])

    monkeypatch.setattr(cli, "_load_store", lambda args: populated_store)
    monkeypatch.setattr(cli, "_load_provider", lambda args: fake_provider)

    out_path = tmp_path / "out.jsonl"

    # First run
    cli.main([
        "--corpus", str(tmp_path / "kb.sqlite"),
        "--provider", "deepseek",
        "--mode", "single",
        "--output", str(out_path),
        "--no-self-consistency",
        "--no-budget-guard",
    ])
    first_lines = out_path.read_text(encoding="utf-8").splitlines()

    # Second run with --resume should add nothing (all chunks already seen)
    cli.main([
        "--corpus", str(tmp_path / "kb.sqlite"),
        "--provider", "deepseek",
        "--mode", "single",
        "--output", str(out_path),
        "--no-self-consistency",
        "--no-budget-guard",
        "--resume",
    ])
    second_lines = out_path.read_text(encoding="utf-8").splitlines()

    assert second_lines == first_lines


def test_cli_budget_guard_aborts_when_estimate_exceeds_cap(tmp_path, populated_store, monkeypatch):
    from scripts import generate_synthetic_qa as cli

    fake_provider = _FakeProvider(responses=["unused"])
    monkeypatch.setattr(cli, "_load_store", lambda args: populated_store)
    monkeypatch.setattr(cli, "_load_provider", lambda args: fake_provider)

    out_path = tmp_path / "out.jsonl"
    with pytest.raises(SystemExit):
        cli.main([
            "--corpus", str(tmp_path / "kb.sqlite"),
            "--provider", "deepseek",
            "--mode", "single",
            "--output", str(out_path),
            "--max-budget-usd", "0.0000001",  # absurdly low
        ])
