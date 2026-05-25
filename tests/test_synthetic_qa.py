"""Tests for app.services.synthetic_qa — pure-logic Q&A generator."""

from __future__ import annotations

import pytest


def test_module_imports():
    """Module imports without side effects."""
    from app.services import synthetic_qa

    assert hasattr(synthetic_qa, "__name__")
    assert synthetic_qa.__name__ == "app.services.synthetic_qa"


def test_qa_pair_to_dict_uses_canonical_fields():
    from app.services.synthetic_qa import QAPair

    pair = QAPair(
        instruction="What is X?",
        input="Context paragraph.",
        output="X is the answer.",
        source_chunk_id=42,
    )

    data = pair.to_dict()
    assert data == {
        "instruction": "What is X?",
        "input": "Context paragraph.",
        "output": "X is the answer.",
        "meta": {"source_chunk_id": 42},
    }


def test_qa_pair_to_jsonl_line_is_single_line():
    from app.services.synthetic_qa import QAPair

    pair = QAPair(
        instruction="Q?",
        input="",
        output="A.",
        source_chunk_id=1,
    )

    line = pair.to_jsonl_line()

    assert line.endswith("\n")
    assert line.count("\n") == 1
    assert "\\n" not in line  # No literal escaped newlines in output values


def test_qa_pair_from_jsonl_line_round_trip():
    from app.services.synthetic_qa import QAPair

    original = QAPair(
        instruction="Q with «русские» symbols?",
        input="ctx",
        output="A.",
        source_chunk_id=7,
    )

    parsed = QAPair.from_jsonl_line(original.to_jsonl_line())

    assert parsed == original


def test_length_filter_accepts_in_range_pair():
    from app.services.synthetic_qa import QAPair, length_ok

    pair = QAPair(
        instruction="What does the regulation say about Y?",
        input="",
        output="The regulation states that Y must be done following X procedure.",
        source_chunk_id=1,
    )

    assert length_ok(pair) is True


def test_length_filter_rejects_short_instruction():
    from app.services.synthetic_qa import QAPair, length_ok

    pair = QAPair(
        instruction="Why?",  # 4 chars < 10
        input="",
        output="A long enough answer goes here to pass that threshold.",
        source_chunk_id=1,
    )

    assert length_ok(pair) is False


def test_length_filter_rejects_long_instruction():
    from app.services.synthetic_qa import QAPair, length_ok

    pair = QAPair(
        instruction="x" * 201,  # 201 chars > 200
        input="",
        output="A long enough answer goes here to pass that threshold.",
        source_chunk_id=1,
    )

    assert length_ok(pair) is False


def test_length_filter_rejects_short_output():
    from app.services.synthetic_qa import QAPair, length_ok

    pair = QAPair(
        instruction="What is the rule about Z?",
        input="",
        output="Short.",  # 6 chars < 30
        source_chunk_id=1,
    )

    assert length_ok(pair) is False


def test_length_filter_rejects_long_output():
    from app.services.synthetic_qa import QAPair, length_ok

    pair = QAPair(
        instruction="What is the rule about Z?",
        input="",
        output="x" * 2001,  # 2001 chars > 2000
        source_chunk_id=1,
    )

    assert length_ok(pair) is False


def test_refusal_filter_accepts_normal_answer():
    from app.services.synthetic_qa import is_refusal

    assert is_refusal("The procedure requires two signatures.") is False


def test_refusal_filter_detects_english_refusal():
    from app.services.synthetic_qa import is_refusal

    assert is_refusal("I cannot answer this question.") is True
    assert is_refusal("Sorry, I can't help with that.") is True
    assert is_refusal("As an AI language model, I cannot...") is True


def test_refusal_filter_detects_russian_refusal():
    from app.services.synthetic_qa import is_refusal

    assert is_refusal("Извините, я не могу ответить на этот вопрос.") is True
    assert is_refusal("Я не имею возможности ответить.") is True
    assert is_refusal("Как языковая модель, я не могу") is True


def test_refusal_filter_is_case_insensitive():
    from app.services.synthetic_qa import is_refusal

    assert is_refusal("I CANNOT answer") is True
    assert is_refusal("извините, я Не Могу") is True


def test_refusal_filter_handles_empty_string():
    from app.services.synthetic_qa import is_refusal

    assert is_refusal("") is False
    assert is_refusal("   ") is False


def test_self_consistency_accepts_identical_text():
    from app.services.synthetic_qa import self_consistent

    text = "The annual leave is 28 calendar days per year."
    assert self_consistent(text, text) is True


def test_self_consistency_accepts_paraphrase():
    from app.services.synthetic_qa import self_consistent

    a = "Annual leave is twenty-eight calendar days each year for every employee."
    b = "Each employee is entitled to twenty-eight calendar days of annual leave per year."
    assert self_consistent(a, b) is True


def test_self_consistency_rejects_unrelated_text():
    from app.services.synthetic_qa import self_consistent

    a = "Annual leave is 28 calendar days per year."
    b = "The kitchen ventilation system needs monthly inspection."
    assert self_consistent(a, b) is False


def test_self_consistency_handles_empty_text():
    from app.services.synthetic_qa import self_consistent

    assert self_consistent("", "") is False
    assert self_consistent("Some text.", "") is False


def test_self_consistency_threshold_can_be_overridden():
    from app.services.synthetic_qa import self_consistent

    a = "alpha beta gamma delta"
    b = "alpha beta zeta theta"  # 2/6 unique tokens shared = 0.33 Jaccard

    assert self_consistent(a, b, threshold=0.5) is False
    assert self_consistent(a, b, threshold=0.3) is True


def test_generation_mode_has_three_values():
    from app.services.synthetic_qa import GenerationMode

    assert GenerationMode.SINGLE.value == "single"
    assert GenerationMode.PARAPHRASE.value == "paraphrase"
    assert GenerationMode.MULTI_HOP.value == "multi-hop"


def test_build_prompt_single_includes_chunk_text():
    from app.services.synthetic_qa import GenerationMode, build_prompt

    chunk_text = "The safety regulation requires a daily inspection."
    prompt = build_prompt(GenerationMode.SINGLE, [chunk_text], chunk_ids=[1])

    assert chunk_text in prompt
    assert "JSON" in prompt
    assert "instruction" in prompt
    assert "output" in prompt


def test_build_prompt_paraphrase_requests_variants():
    from app.services.synthetic_qa import GenerationMode, build_prompt

    prompt = build_prompt(GenerationMode.PARAPHRASE, ["abc"], chunk_ids=[1])

    assert "3" in prompt or "три" in prompt.lower()
    assert "paraphr" in prompt.lower() or "перефраз" in prompt.lower()


def test_build_prompt_multi_hop_requires_multiple_chunks():
    from app.services.synthetic_qa import GenerationMode, build_prompt

    chunks = ["alpha section", "beta section", "gamma section"]
    prompt = build_prompt(GenerationMode.MULTI_HOP, chunks, chunk_ids=[1, 2, 3])

    for chunk in chunks:
        assert chunk in prompt
    assert "multi" in prompt.lower() or "несколько" in prompt.lower()


def test_build_prompt_multi_hop_with_single_chunk_raises():
    import pytest as _pytest
    from app.services.synthetic_qa import GenerationMode, build_prompt

    with _pytest.raises(ValueError):
        build_prompt(GenerationMode.MULTI_HOP, ["only one"], chunk_ids=[1])


def test_parse_response_clean_object():
    from app.services.synthetic_qa import parse_qa_response

    raw = '{"instruction": "Q?", "input": "", "output": "A. [doc_chunk:1]"}'
    pairs = parse_qa_response(raw, source_chunk_id=1)

    assert len(pairs) == 1
    assert pairs[0].instruction == "Q?"
    assert pairs[0].output == "A. [doc_chunk:1]"
    assert pairs[0].source_chunk_id == 1


def test_parse_response_clean_array():
    from app.services.synthetic_qa import parse_qa_response

    raw = (
        '[{"instruction":"Q1","input":"","output":"A [doc_chunk:5]"},'
        '{"instruction":"Q2","input":"","output":"A [doc_chunk:5]"}]'
    )
    pairs = parse_qa_response(raw, source_chunk_id=5)

    assert len(pairs) == 2
    assert pairs[0].instruction == "Q1"
    assert pairs[1].instruction == "Q2"


def test_parse_response_strips_markdown_fence():
    from app.services.synthetic_qa import parse_qa_response

    raw = (
        "```json\n"
        '{"instruction": "Q?", "input": "", "output": "A. [doc_chunk:2]"}\n'
        "```"
    )
    pairs = parse_qa_response(raw, source_chunk_id=2)

    assert len(pairs) == 1
    assert pairs[0].instruction == "Q?"


def test_parse_response_recovers_first_json_object():
    from app.services.synthetic_qa import parse_qa_response

    raw = (
        "Вот результат:\n"
        '{"instruction": "Q?", "input": "", "output": "A. [doc_chunk:3]"}\n'
        "Надеюсь подойдёт!"
    )
    pairs = parse_qa_response(raw, source_chunk_id=3)

    assert len(pairs) == 1


def test_parse_response_returns_empty_on_malformed():
    from app.services.synthetic_qa import parse_qa_response

    pairs = parse_qa_response("This is not JSON at all", source_chunk_id=1)
    assert pairs == []


def test_parse_response_skips_items_missing_required_fields():
    from app.services.synthetic_qa import parse_qa_response

    raw = (
        '[{"instruction":"Q1","input":"","output":"A"},'
        '{"instruction":"Q2","input":""},'  # missing output
        '{"output":"A only","input":""}]'  # missing instruction
    )
    pairs = parse_qa_response(raw, source_chunk_id=9)

    assert len(pairs) == 1
    assert pairs[0].instruction == "Q1"


def test_estimate_chunk_cost_for_known_provider():
    from app.services.synthetic_qa import GenerationMode, estimate_chunk_cost_usd

    chunk_text = "x" * 4000  # ≈1000 input tokens
    cost = estimate_chunk_cost_usd(
        provider="deepseek",
        model="deepseek-chat",
        mode=GenerationMode.SINGLE,
        chunk_chars=len(chunk_text),
    )

    # DeepSeek-chat is cheap; one chunk single mode should be far below 1c
    assert 0 < cost < 0.01


def test_estimate_chunk_cost_higher_for_paraphrase():
    from app.services.synthetic_qa import GenerationMode, estimate_chunk_cost_usd

    chunk_chars = 4000
    single = estimate_chunk_cost_usd("deepseek", "deepseek-chat", GenerationMode.SINGLE, chunk_chars)
    paraphrase = estimate_chunk_cost_usd("deepseek", "deepseek-chat", GenerationMode.PARAPHRASE, chunk_chars)

    assert paraphrase > single


def test_estimate_chunk_cost_unknown_provider_returns_none():
    from app.services.synthetic_qa import GenerationMode, estimate_chunk_cost_usd

    cost = estimate_chunk_cost_usd("unicorn-llm", "model-x", GenerationMode.SINGLE, 4000)
    assert cost is None


def test_estimate_total_cost_sums_chunks():
    from app.services.synthetic_qa import GenerationMode, estimate_total_cost_usd

    chunk_chars = [4000, 4000, 8000]
    total = estimate_total_cost_usd(
        provider="deepseek",
        model="deepseek-chat",
        mode=GenerationMode.SINGLE,
        chunk_chars=chunk_chars,
    )

    assert total is not None
    assert total > 0


class _FakeProvider:
    """Test double matching the protocol used by SyntheticQAGenerator."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    def generate(self, prompt, *, system=None, max_tokens=None, temperature=None):
        from app.services.kb_llm import LLMResponse

        self.calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        text = self._responses.pop(0)
        return LLMResponse(text=text, provider="fake", model="fake-model", elapsed_ms=1.0)


def test_generator_single_mode_returns_one_pair():
    from app.services.synthetic_qa import (
        GenerationMode,
        SyntheticQAGenerator,
    )

    provider = _FakeProvider(
        responses=[
            '{"instruction":"What is the rule about Y?","input":"",'
            '"output":"The rule states that Y must follow X procedure with care taken to verify compliance. [doc_chunk:7]"}',
            # Second call for self-consistency
            '{"instruction":"What does the rule say about Y?","input":"",'
            '"output":"Y must follow X procedure with verification of compliance. [doc_chunk:7]"}',
        ]
    )
    generator = SyntheticQAGenerator(provider=provider)

    pairs = generator.generate_for_chunk(
        chunks=["The rule says Y must follow X procedure with compliance check."],
        chunk_ids=[7],
        mode=GenerationMode.SINGLE,
    )

    assert len(pairs) == 1
    assert pairs[0].source_chunk_id == 7


def test_generator_skips_refusal_response():
    from app.services.synthetic_qa import (
        GenerationMode,
        SyntheticQAGenerator,
    )

    provider = _FakeProvider(
        responses=['{"instruction":"Q?","input":"","output":"I cannot answer this question, sorry."}']
    )
    generator = SyntheticQAGenerator(provider=provider)

    pairs = generator.generate_for_chunk(
        chunks=["some text"], chunk_ids=[1], mode=GenerationMode.SINGLE
    )

    assert pairs == []


def test_generator_drops_pairs_failing_length_filter():
    from app.services.synthetic_qa import (
        GenerationMode,
        SyntheticQAGenerator,
    )

    provider = _FakeProvider(
        responses=['{"instruction":"Q?","input":"","output":"too short"}']  # output 9 chars < 30
    )
    generator = SyntheticQAGenerator(provider=provider)

    pairs = generator.generate_for_chunk(
        chunks=["some text"], chunk_ids=[1], mode=GenerationMode.SINGLE
    )

    assert pairs == []


def test_generator_drops_pairs_failing_self_consistency():
    from app.services.synthetic_qa import (
        GenerationMode,
        SyntheticQAGenerator,
    )

    provider = _FakeProvider(
        responses=[
            # First generation
            '{"instruction":"What is the rule about safety in the workplace?","input":"",'
            '"output":"The rule requires safety helmets at all times in production areas. [doc_chunk:1]"}',
            # Second generation - completely unrelated content
            '{"instruction":"What is the kitchen schedule?","input":"",'
            '"output":"Lunch is served between twelve and one thirty in the canteen building. [doc_chunk:1]"}',
        ]
    )
    generator = SyntheticQAGenerator(provider=provider)

    pairs = generator.generate_for_chunk(
        chunks=["chunk text"], chunk_ids=[1], mode=GenerationMode.SINGLE
    )

    assert pairs == []


def test_generator_can_disable_self_consistency():
    from app.services.synthetic_qa import (
        GenerationMode,
        SyntheticQAGenerator,
    )

    provider = _FakeProvider(
        responses=[
            '{"instruction":"What is the rule about Y?","input":"",'
            '"output":"The rule states Y must follow procedure X with verification. [doc_chunk:1]"}',
        ]
    )
    generator = SyntheticQAGenerator(
        provider=provider, check_self_consistency=False
    )

    pairs = generator.generate_for_chunk(
        chunks=["text"], chunk_ids=[1], mode=GenerationMode.SINGLE
    )

    # Without self-consistency, only one provider call happens
    assert len(provider.calls) == 1
    assert len(pairs) == 1


def test_load_processed_chunk_ids_from_missing_file(tmp_path):
    from app.services.synthetic_qa import load_processed_chunk_ids

    processed = load_processed_chunk_ids(tmp_path / "missing.jsonl")
    assert processed == set()


def test_load_processed_chunk_ids_reads_meta(tmp_path):
    from app.services.synthetic_qa import QAPair, load_processed_chunk_ids

    path = tmp_path / "out.jsonl"
    pairs = [
        QAPair(instruction="Q1", input="", output="A1 long enough text here", source_chunk_id=10),
        QAPair(instruction="Q2", input="", output="A2 long enough text here", source_chunk_id=20),
        QAPair(instruction="Q3", input="", output="A3 long enough text here", source_chunk_id=10),  # dup
    ]
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(pair.to_jsonl_line())

    processed = load_processed_chunk_ids(path)
    assert processed == {10, 20}


def test_load_processed_chunk_ids_skips_lines_without_meta(tmp_path):
    from app.services.synthetic_qa import load_processed_chunk_ids

    path = tmp_path / "out.jsonl"
    path.write_text(
        '{"instruction":"Q","input":"","output":"A long enough text goes here for sure"}\n'  # no meta
        '{"instruction":"Q","input":"","output":"A long enough text goes here for sure","meta":{"source_chunk_id":5}}\n',
        encoding="utf-8",
    )

    processed = load_processed_chunk_ids(path)
    assert processed == {5}


def test_load_processed_chunk_ids_tolerates_malformed_lines(tmp_path):
    from app.services.synthetic_qa import QAPair, load_processed_chunk_ids

    path = tmp_path / "out.jsonl"
    pair = QAPair(instruction="Q", input="", output="A long enough text goes here for sure", source_chunk_id=99)
    path.write_text(
        "this is not json\n"
        + pair.to_jsonl_line(),
        encoding="utf-8",
    )

    processed = load_processed_chunk_ids(path)
    assert processed == {99}


def test_jsonl_output_is_consumed_by_validate_dataset(tmp_path):
    """End-to-end: QAPair JSONL must parse via validate_dataset.load_examples."""
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

    from app.services.synthetic_qa import QAPair
    from scripts import validate_dataset as vd

    pairs = [
        QAPair(
            instruction=f"What is rule {i}?",
            input="",
            output=f"Rule {i} states that the corresponding procedure must be followed. [doc_chunk:{i}]",
            source_chunk_id=i,
        )
        for i in range(1, 6)
    ]
    path = tmp_path / "dataset.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(pair.to_jsonl_line())

    loaded = vd.load_examples(path)
    assert len(loaded) == 5
    for i, example in enumerate(loaded, start=1):
        assert example.instruction == f"What is rule {i}?"
        assert example.output.startswith(f"Rule {i}")
