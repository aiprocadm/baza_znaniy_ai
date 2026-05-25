# W1 Synthetic Q&A Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/generate_synthetic_qa.py` that turns an existing KB corpus into a JSONL training dataset by querying a teacher LLM (DeepSeek by default), with three generation modes, quality filters, cost guard, and resume support. Output must validate cleanly through `scripts/validate_dataset.py`.

**Architecture:** Split into a pure-logic module `app/services/synthetic_qa.py` (testable, no I/O) and a CLI wrapper `scripts/generate_synthetic_qa.py` (argparse + file I/O). Reuse existing `app/services/kb_llm.py` (provider chain) and `app/services/kb_store.py` (chunk iteration). The generator class accepts an injected provider, making tests deterministic with fake providers.

**Tech Stack:** Python 3.12, pytest, dataclasses, argparse, json, existing `OpenAICompatibleProvider`, existing `KnowledgeBaseStore`.

**Spec reference:** Workstream W1 in `docs/superpowers/specs/2026-05-25-ml-strengthening-pack-b-design.md`.

---

## File Structure

**Create:**
- `app/services/synthetic_qa.py` — pure logic: `QAPair`, `GenerationMode`, filters, `SyntheticQAGenerator`, cost estimation, JSONL I/O helpers
- `scripts/generate_synthetic_qa.py` — CLI wrapper around the module
- `tests/test_synthetic_qa.py` — unit tests for the module (mocked provider)
- `tests/scripts/test_generate_synthetic_qa.py` — CLI smoke test

**Modify:**
- `README.md` — add usage example at the end of the "обучение LoRA" section

**Dependencies:**
- All existing — no new PyPI packages required for W1 (existing `httpx` covers HTTP)

---

## Task 1: Project skeleton and module imports

**Files:**
- Create: `app/services/synthetic_qa.py`
- Create: `tests/test_synthetic_qa.py`

- [ ] **Step 1.1: Write failing import test**

Create `tests/test_synthetic_qa.py`:

```python
"""Tests for app.services.synthetic_qa — pure-logic Q&A generator."""

from __future__ import annotations

import pytest


def test_module_imports():
    """Module imports without side effects."""
    from app.services import synthetic_qa

    assert hasattr(synthetic_qa, "__name__")
    assert synthetic_qa.__name__ == "app.services.synthetic_qa"
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_synthetic_qa.py::test_module_imports -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.synthetic_qa'`

- [ ] **Step 1.3: Create minimal module**

Create `app/services/synthetic_qa.py`:

```python
"""Generate synthetic supervised Q&A datasets from a KB corpus.

This module is the pure-logic core of Workstream 1 (Synthetic Data
Generation) in the Pack B++ ML strengthening plan. A teacher LLM is
prompted with document chunks and asked to produce diverse Q&A pairs.
The CLI wrapper is in ``scripts/generate_synthetic_qa.py``.

The module is intentionally I/O free: all dependencies (LLM provider,
chunk source) are injected, making the logic deterministic in tests.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

__all__: list[str] = []
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_synthetic_qa.py::test_module_imports -v`
Expected: PASS

- [ ] **Step 1.5: Commit**

```bash
git add app/services/synthetic_qa.py tests/test_synthetic_qa.py
git commit -m "feat(synthetic-qa): add module skeleton for W1"
```

---

## Task 2: QAPair dataclass with JSONL-compatible serialisation

**Files:**
- Modify: `app/services/synthetic_qa.py`
- Modify: `tests/test_synthetic_qa.py`

`validate_dataset.py` expects fields `instruction`, `input`, `output` (alternatives `prompt/question`, `context/background`, `response/answer`). We standardise on `instruction`, `input`, `output` since `train_lora.py` uses the same.

- [ ] **Step 2.1: Write failing tests**

Append to `tests/test_synthetic_qa.py`:

```python
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
```

- [ ] **Step 2.2: Run tests to verify failure**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v`
Expected: 3 tests FAIL with `ImportError` or `AttributeError: module ... has no attribute 'QAPair'`

- [ ] **Step 2.3: Implement QAPair**

Replace the contents of `app/services/synthetic_qa.py` with:

```python
"""Generate synthetic supervised Q&A datasets from a KB corpus."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QAPair:
    """One instruction/input/output triple ready for SFT training.

    Fields match the canonical layout consumed by
    ``scripts/train_lora.py`` and ``scripts/validate_dataset.py``.
    ``source_chunk_id`` is preserved in the ``meta`` sidecar so the
    pipeline can resume by recognising which chunks were already
    processed.
    """

    instruction: str
    input: str
    output: str
    source_chunk_id: int

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "meta": {"source_chunk_id": int(self.source_chunk_id)},
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"

    @classmethod
    def from_jsonl_line(cls, line: str) -> "QAPair":
        data = json.loads(line)
        meta = data.get("meta") or {}
        return cls(
            instruction=str(data["instruction"]),
            input=str(data.get("input", "")),
            output=str(data["output"]),
            source_chunk_id=int(meta.get("source_chunk_id", 0)),
        )


__all__ = ["QAPair"]
```

- [ ] **Step 2.4: Run tests to verify pass**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v`
Expected: All 4 tests PASS

- [ ] **Step 2.5: Commit**

```bash
git add app/services/synthetic_qa.py tests/test_synthetic_qa.py
git commit -m "feat(synthetic-qa): add QAPair dataclass with JSONL round-trip"
```

---

## Task 3: Length filter

**Files:**
- Modify: `app/services/synthetic_qa.py`
- Modify: `tests/test_synthetic_qa.py`

Spec acceptance: instruction 10-200 chars, output 30-2000 chars. Input is optional and unconstrained.

- [ ] **Step 3.1: Write failing tests**

Append to `tests/test_synthetic_qa.py`:

```python
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
```

- [ ] **Step 3.2: Run tests to verify failure**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v -k length_filter`
Expected: 5 tests FAIL with `ImportError`

- [ ] **Step 3.3: Implement length_ok**

Add to `app/services/synthetic_qa.py` (above `__all__`):

```python
MIN_INSTRUCTION_CHARS = 10
MAX_INSTRUCTION_CHARS = 200
MIN_OUTPUT_CHARS = 30
MAX_OUTPUT_CHARS = 2000


def length_ok(pair: QAPair) -> bool:
    """Return True when *pair* is within configured length bounds.

    Bounds come from the W1 acceptance criteria in
    docs/superpowers/specs/2026-05-25-ml-strengthening-pack-b-design.md.
    """

    instruction = pair.instruction.strip()
    output = pair.output.strip()
    if not (MIN_INSTRUCTION_CHARS <= len(instruction) <= MAX_INSTRUCTION_CHARS):
        return False
    if not (MIN_OUTPUT_CHARS <= len(output) <= MAX_OUTPUT_CHARS):
        return False
    return True
```

Update `__all__`:

```python
__all__ = ["QAPair", "length_ok", "MIN_INSTRUCTION_CHARS", "MAX_INSTRUCTION_CHARS", "MIN_OUTPUT_CHARS", "MAX_OUTPUT_CHARS"]
```

- [ ] **Step 3.4: Run tests to verify pass**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v`
Expected: All tests so far PASS (4 from Task 2 + 5 from Task 3 = 9)

- [ ] **Step 3.5: Commit**

```bash
git add app/services/synthetic_qa.py tests/test_synthetic_qa.py
git commit -m "feat(synthetic-qa): add length filter for Q&A pairs"
```

---

## Task 4: Refusal filter

**Files:**
- Modify: `app/services/synthetic_qa.py`
- Modify: `tests/test_synthetic_qa.py`

Teacher LLMs sometimes refuse to answer. We detect Russian and English refusal phrases and drop the pair.

- [ ] **Step 4.1: Write failing tests**

Append to `tests/test_synthetic_qa.py`:

```python
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
```

- [ ] **Step 4.2: Run tests to verify failure**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v -k refusal`
Expected: 5 tests FAIL with `ImportError`

- [ ] **Step 4.3: Implement is_refusal**

Add to `app/services/synthetic_qa.py` (after the length section):

```python
_REFUSAL_MARKERS = (
    # English
    "i cannot answer",
    "i can't answer",
    "i can't help",
    "i cannot help",
    "as an ai language model",
    "i am not able to",
    "i'm not able to",
    "i'm sorry, but i can't",
    "sorry, i can't",
    "sorry, i cannot",
    # Russian
    "извините, я не могу",
    "я не могу ответить",
    "я не имею возможности",
    "как языковая модель, я не могу",
    "к сожалению, я не могу",
)


def is_refusal(text: str) -> bool:
    """Return True when *text* looks like a generic teacher refusal."""

    if not text or not text.strip():
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)
```

Update `__all__`:

```python
__all__ = [
    "QAPair",
    "length_ok",
    "is_refusal",
    "MIN_INSTRUCTION_CHARS",
    "MAX_INSTRUCTION_CHARS",
    "MIN_OUTPUT_CHARS",
    "MAX_OUTPUT_CHARS",
]
```

- [ ] **Step 4.4: Run tests to verify pass**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v`
Expected: All 14 tests PASS

- [ ] **Step 4.5: Commit**

```bash
git add app/services/synthetic_qa.py tests/test_synthetic_qa.py
git commit -m "feat(synthetic-qa): add bilingual refusal detector"
```

---

## Task 5: Self-consistency check via token Jaccard similarity

**Files:**
- Modify: `app/services/synthetic_qa.py`
- Modify: `tests/test_synthetic_qa.py`

Two generations of the same Q on the same chunk should overlap heavily. We use a lightweight Jaccard similarity on lowercased word tokens — no embedding model dependency. Threshold of 0.4 catches paraphrases (high similarity) while rejecting unrelated answers (low similarity).

- [ ] **Step 5.1: Write failing tests**

Append to `tests/test_synthetic_qa.py`:

```python
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
```

- [ ] **Step 5.2: Run tests to verify failure**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v -k self_consist`
Expected: 5 tests FAIL with `ImportError`

- [ ] **Step 5.3: Implement self_consistent**

Add to `app/services/synthetic_qa.py` (after the refusal section):

```python
import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
DEFAULT_CONSISTENCY_THRESHOLD = 0.4


def _tokenise(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def self_consistent(
    text_a: str,
    text_b: str,
    *,
    threshold: float = DEFAULT_CONSISTENCY_THRESHOLD,
) -> bool:
    """Return True when two generated answers overlap enough to trust.

    Computes a lowercase-token Jaccard similarity. ``threshold`` defaults
    to 0.4 — empirically high enough to catch paraphrases on the same
    chunk while rejecting unrelated content. Either text being empty is
    treated as failure.
    """

    if not text_a.strip() or not text_b.strip():
        return False
    tokens_a = _tokenise(text_a)
    tokens_b = _tokenise(text_b)
    if not tokens_a or not tokens_b:
        return False
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    similarity = intersection / union
    return similarity >= threshold
```

Move the `import re` to the top of the file with the other stdlib imports. Update `__all__`:

```python
__all__ = [
    "QAPair",
    "length_ok",
    "is_refusal",
    "self_consistent",
    "MIN_INSTRUCTION_CHARS",
    "MAX_INSTRUCTION_CHARS",
    "MIN_OUTPUT_CHARS",
    "MAX_OUTPUT_CHARS",
    "DEFAULT_CONSISTENCY_THRESHOLD",
]
```

- [ ] **Step 5.4: Run tests to verify pass**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v`
Expected: All 19 tests PASS

- [ ] **Step 5.5: Commit**

```bash
git add app/services/synthetic_qa.py tests/test_synthetic_qa.py
git commit -m "feat(synthetic-qa): add Jaccard-based self-consistency check"
```

---

## Task 6: GenerationMode enum and prompt templates

**Files:**
- Modify: `app/services/synthetic_qa.py`
- Modify: `tests/test_synthetic_qa.py`

Three modes per spec: SINGLE (one Q&A per chunk), PARAPHRASE (3 question variants for the same answer), MULTI_HOP (a question that requires combining several chunks). Each mode has its own prompt template.

- [ ] **Step 6.1: Write failing tests**

Append to `tests/test_synthetic_qa.py`:

```python
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
```

- [ ] **Step 6.2: Run tests to verify failure**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v -k "generation_mode or build_prompt"`
Expected: 5 tests FAIL with `ImportError`

- [ ] **Step 6.3: Implement GenerationMode and build_prompt**

Add to `app/services/synthetic_qa.py` (after the self_consistent section):

```python
from enum import Enum


class GenerationMode(str, Enum):
    SINGLE = "single"
    PARAPHRASE = "paraphrase"
    MULTI_HOP = "multi-hop"


_PROMPT_SINGLE = (
    "Ты — эксперт по составлению обучающих примеров для AI-помощника по "
    "корпоративным документам. На основе фрагмента документа сгенерируй "
    "ОДИН вопрос, который мог бы задать сотрудник компании, и точный "
    "ответ. Ответ должен опираться только на фрагмент и заканчиваться "
    "указанием источника в формате [doc_chunk:{chunk_id}]."
    "\n\n"
    "Фрагмент [doc_chunk:{chunk_id}]:\n{chunk_text}\n\n"
    "Верни строго JSON без дополнительного текста:\n"
    '{{"instruction": "<вопрос>", "input": "", '
    '"output": "<ответ> [doc_chunk:{chunk_id}]"}}'
)

_PROMPT_PARAPHRASE = (
    "Ты — эксперт по составлению обучающих примеров. На основе "
    "фрагмента документа сгенерируй ТРИ разных перефразирования одного "
    "и того же вопроса и общий ответ, опирающийся на фрагмент. Вопросы "
    "должны различаться по формулировке, но иметь один и тот же смысл."
    "\n\n"
    "Фрагмент [doc_chunk:{chunk_id}]:\n{chunk_text}\n\n"
    "Верни строго JSON-массив без дополнительного текста:\n"
    "[\n"
    '  {{"instruction": "<вопрос 1>", "input": "", '
    '"output": "<общий ответ> [doc_chunk:{chunk_id}]"}},\n'
    '  {{"instruction": "<вопрос 2 — paraphrase>", "input": "", '
    '"output": "<тот же ответ> [doc_chunk:{chunk_id}]"}},\n'
    '  {{"instruction": "<вопрос 3 — paraphrase>", "input": "", '
    '"output": "<тот же ответ> [doc_chunk:{chunk_id}]"}}\n'
    "]"
)

_PROMPT_MULTI_HOP = (
    "Ты — эксперт по составлению обучающих примеров. Тебе даны "
    "{n_chunks} фрагментов из разных мест документа. Сгенерируй ОДИН "
    "вопрос, ответ на который требует объединения информации из всех "
    "приведённых фрагментов (multi-hop). Ответ должен опираться на "
    "комбинацию фрагментов и перечислить источники."
    "\n\n"
    "{chunks_block}\n"
    "Верни строго JSON без дополнительного текста:\n"
    '{{"instruction": "<вопрос, требующий объединения>", "input": "", '
    '"output": "<ответ с указанием [doc_chunk:X] для каждого использованного фрагмента>"}}'
)


def build_prompt(
    mode: GenerationMode,
    chunks: list[str],
    *,
    chunk_ids: list[int],
) -> str:
    """Return the teacher prompt for *mode*.

    ``chunks`` and ``chunk_ids`` must align (same length, same order).
    ``MULTI_HOP`` requires at least 2 chunks; raises ``ValueError``
    otherwise.
    """

    if len(chunks) != len(chunk_ids):
        raise ValueError("chunks and chunk_ids must have equal length")
    if not chunks:
        raise ValueError("at least one chunk is required")

    if mode is GenerationMode.SINGLE:
        return _PROMPT_SINGLE.format(
            chunk_text=chunks[0], chunk_id=chunk_ids[0]
        )

    if mode is GenerationMode.PARAPHRASE:
        return _PROMPT_PARAPHRASE.format(
            chunk_text=chunks[0], chunk_id=chunk_ids[0]
        )

    if mode is GenerationMode.MULTI_HOP:
        if len(chunks) < 2:
            raise ValueError("multi-hop mode requires at least 2 chunks")
        block = "\n\n".join(
            f"Фрагмент [doc_chunk:{cid}]:\n{text}"
            for text, cid in zip(chunks, chunk_ids)
        )
        return _PROMPT_MULTI_HOP.format(
            n_chunks=len(chunks), chunks_block=block
        )

    raise ValueError(f"Unsupported generation mode: {mode!r}")
```

Update `__all__`:

```python
__all__ = [
    "QAPair",
    "GenerationMode",
    "length_ok",
    "is_refusal",
    "self_consistent",
    "build_prompt",
    "MIN_INSTRUCTION_CHARS",
    "MAX_INSTRUCTION_CHARS",
    "MIN_OUTPUT_CHARS",
    "MAX_OUTPUT_CHARS",
    "DEFAULT_CONSISTENCY_THRESHOLD",
]
```

- [ ] **Step 6.4: Run tests to verify pass**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v`
Expected: All 24 tests PASS

- [ ] **Step 6.5: Commit**

```bash
git add app/services/synthetic_qa.py tests/test_synthetic_qa.py
git commit -m "feat(synthetic-qa): add GenerationMode and prompt templates"
```

---

## Task 7: JSON response parser

**Files:**
- Modify: `app/services/synthetic_qa.py`
- Modify: `tests/test_synthetic_qa.py`

Teacher responses can be: clean JSON, JSON wrapped in markdown code fence (```json ... ```), JSON with extra text before/after. We need a tolerant parser.

- [ ] **Step 7.1: Write failing tests**

Append to `tests/test_synthetic_qa.py`:

```python
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
```

- [ ] **Step 7.2: Run tests to verify failure**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v -k parse_response`
Expected: 6 tests FAIL with `ImportError`

- [ ] **Step 7.3: Implement parse_qa_response**

Add to `app/services/synthetic_qa.py` (after the `build_prompt` section):

```python
_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    match = _FENCE_PATTERN.match(text)
    return match.group(1) if match else text


def _extract_first_json_payload(text: str) -> str | None:
    """Return the first top-level JSON object or array substring in *text*."""

    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == open_ch:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == close_ch and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : i + 1]
    return None


def parse_qa_response(raw: str, *, source_chunk_id: int) -> list[QAPair]:
    """Parse a teacher response into zero or more :class:`QAPair` objects.

    Tolerates markdown code fences and surrounding prose. Items missing
    ``instruction`` or ``output`` are dropped silently. Returns an empty
    list on unrecoverable malformed input.
    """

    if not raw or not raw.strip():
        return []

    candidate = _strip_markdown_fence(raw).strip()

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        extracted = _extract_first_json_payload(candidate)
        if extracted is None:
            return []
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        items: list[dict[str, object]] = [data]
    elif isinstance(data, list):
        items = [item for item in data if isinstance(item, dict)]
    else:
        return []

    pairs: list[QAPair] = []
    for item in items:
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()
        if not instruction or not output:
            continue
        input_text = str(item.get("input", "")).strip()
        pairs.append(
            QAPair(
                instruction=instruction,
                input=input_text,
                output=output,
                source_chunk_id=int(source_chunk_id),
            )
        )

    return pairs
```

Update `__all__`:

```python
__all__ = [
    "QAPair",
    "GenerationMode",
    "length_ok",
    "is_refusal",
    "self_consistent",
    "build_prompt",
    "parse_qa_response",
    "MIN_INSTRUCTION_CHARS",
    "MAX_INSTRUCTION_CHARS",
    "MIN_OUTPUT_CHARS",
    "MAX_OUTPUT_CHARS",
    "DEFAULT_CONSISTENCY_THRESHOLD",
]
```

- [ ] **Step 7.4: Run tests to verify pass**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v`
Expected: All 30 tests PASS

- [ ] **Step 7.5: Commit**

```bash
git add app/services/synthetic_qa.py tests/test_synthetic_qa.py
git commit -m "feat(synthetic-qa): add tolerant JSON response parser"
```

---

## Task 8: Cost estimator

**Files:**
- Modify: `app/services/synthetic_qa.py`
- Modify: `tests/test_synthetic_qa.py`

We use a static pricing table (input/output USD per 1M tokens) per known provider+model. Approximate cost = sum(input_tokens × in_price + output_tokens × out_price). Token count approximated as `len(text) / 4` for English+Russian mix (good enough for a guard).

- [ ] **Step 8.1: Write failing tests**

Append to `tests/test_synthetic_qa.py`:

```python
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
```

- [ ] **Step 8.2: Run tests to verify failure**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v -k estimate`
Expected: 4 tests FAIL with `ImportError`

- [ ] **Step 8.3: Implement cost estimator**

Add to `app/services/synthetic_qa.py` (after `parse_qa_response`):

```python
# USD per 1M tokens. Conservative figures from late-2024 public pricing;
# refresh when a provider publishes new rates. Unknown (provider, model)
# combinations return None so the CLI can disable the budget guard with
# a clear warning rather than miscalculate silently.
_PRICING_USD_PER_M: dict[tuple[str, str], tuple[float, float]] = {
    ("deepseek", "deepseek-chat"): (0.014, 0.28),
    ("groq", "llama-3.3-70b-versatile"): (0.59, 0.79),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openrouter", "deepseek/deepseek-chat"): (0.014, 0.28),
}

# Approximate output token budget per generation mode. Used together
# with input-token estimates from chunk size to bound cost forecasts.
_OUTPUT_TOKENS_PER_MODE: dict[GenerationMode, int] = {
    GenerationMode.SINGLE: 200,
    GenerationMode.PARAPHRASE: 500,
    GenerationMode.MULTI_HOP: 350,
}

CHARS_PER_TOKEN_HEURISTIC = 4.0


def _chars_to_tokens(chars: int) -> int:
    return max(1, int(chars / CHARS_PER_TOKEN_HEURISTIC))


def estimate_chunk_cost_usd(
    provider: str,
    model: str,
    mode: GenerationMode,
    chunk_chars: int,
) -> float | None:
    """Estimated dollar cost of generating one batch for one chunk.

    Returns ``None`` when the (provider, model) tuple is not in the
    pricing table; the CLI then warns and disables the budget guard.
    """

    pricing = _PRICING_USD_PER_M.get((provider, model))
    if pricing is None:
        return None
    input_price, output_price = pricing

    input_tokens = _chars_to_tokens(chunk_chars) + 200  # 200 ≈ prompt overhead
    output_tokens = _OUTPUT_TOKENS_PER_MODE[mode]

    cost = (
        input_tokens * input_price / 1_000_000
        + output_tokens * output_price / 1_000_000
    )
    return cost


def estimate_total_cost_usd(
    *,
    provider: str,
    model: str,
    mode: GenerationMode,
    chunk_chars: list[int],
) -> float | None:
    """Sum of :func:`estimate_chunk_cost_usd` across all chunks."""

    total = 0.0
    for chars in chunk_chars:
        per_chunk = estimate_chunk_cost_usd(provider, model, mode, chars)
        if per_chunk is None:
            return None
        total += per_chunk
    return total
```

Update `__all__`:

```python
__all__ = [
    "QAPair",
    "GenerationMode",
    "length_ok",
    "is_refusal",
    "self_consistent",
    "build_prompt",
    "parse_qa_response",
    "estimate_chunk_cost_usd",
    "estimate_total_cost_usd",
    "MIN_INSTRUCTION_CHARS",
    "MAX_INSTRUCTION_CHARS",
    "MIN_OUTPUT_CHARS",
    "MAX_OUTPUT_CHARS",
    "DEFAULT_CONSISTENCY_THRESHOLD",
    "CHARS_PER_TOKEN_HEURISTIC",
]
```

- [ ] **Step 8.4: Run tests to verify pass**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v`
Expected: All 34 tests PASS

- [ ] **Step 8.5: Commit**

```bash
git add app/services/synthetic_qa.py tests/test_synthetic_qa.py
git commit -m "feat(synthetic-qa): add per-chunk and total cost estimators"
```

---

## Task 9: Generator class with injected provider (mocked LLM)

**Files:**
- Modify: `app/services/synthetic_qa.py`
- Modify: `tests/test_synthetic_qa.py`

The generator wraps the filter+parse+self-consistency pipeline behind one `generate_for_chunk()` method. We inject an `LLMProvider` protocol so tests can use a fake without touching the network. The protocol matches `OpenAICompatibleProvider.generate()`'s shape.

- [ ] **Step 9.1: Write failing tests**

Append to `tests/test_synthetic_qa.py`:

```python
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
```

- [ ] **Step 9.2: Run tests to verify failure**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v -k generator`
Expected: 5 tests FAIL with `ImportError`

- [ ] **Step 9.3: Implement SyntheticQAGenerator**

Add to `app/services/synthetic_qa.py` (after the cost section):

```python
from typing import Protocol


class LLMProvider(Protocol):
    """Subset of ``OpenAICompatibleProvider`` used by the generator."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ): ...


@dataclass(slots=True)
class SyntheticQAGenerator:
    """Pipeline: prompt → teacher → parse → filter → optional self-check.

    The generator stays pure: no file I/O, no env reads, no global
    state. All side effects belong in the CLI wrapper.
    """

    provider: LLMProvider
    check_self_consistency: bool = True
    consistency_threshold: float = DEFAULT_CONSISTENCY_THRESHOLD
    max_output_tokens: int = 800
    temperature: float = 0.4

    def generate_for_chunk(
        self,
        *,
        chunks: list[str],
        chunk_ids: list[int],
        mode: GenerationMode,
    ) -> list[QAPair]:
        prompt = build_prompt(mode, chunks, chunk_ids=chunk_ids)
        first = self._call_provider(prompt)
        candidates = parse_qa_response(first, source_chunk_id=chunk_ids[0])
        candidates = [p for p in candidates if not is_refusal(p.output)]
        candidates = [p for p in candidates if length_ok(p)]

        if not self.check_self_consistency or not candidates:
            return candidates

        second = self._call_provider(prompt)
        second_candidates = parse_qa_response(second, source_chunk_id=chunk_ids[0])
        if not second_candidates:
            return []

        kept: list[QAPair] = []
        for pair in candidates:
            for other in second_candidates:
                if self_consistent(
                    pair.output,
                    other.output,
                    threshold=self.consistency_threshold,
                ):
                    kept.append(pair)
                    break
        return kept

    def _call_provider(self, prompt: str) -> str:
        response = self.provider.generate(
            prompt,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )
        return getattr(response, "text", "") or ""
```

Update `__all__`:

```python
__all__ = [
    "QAPair",
    "GenerationMode",
    "LLMProvider",
    "SyntheticQAGenerator",
    "length_ok",
    "is_refusal",
    "self_consistent",
    "build_prompt",
    "parse_qa_response",
    "estimate_chunk_cost_usd",
    "estimate_total_cost_usd",
    "MIN_INSTRUCTION_CHARS",
    "MAX_INSTRUCTION_CHARS",
    "MIN_OUTPUT_CHARS",
    "MAX_OUTPUT_CHARS",
    "DEFAULT_CONSISTENCY_THRESHOLD",
    "CHARS_PER_TOKEN_HEURISTIC",
]
```

- [ ] **Step 9.4: Run tests to verify pass**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v`
Expected: All 39 tests PASS

- [ ] **Step 9.5: Commit**

```bash
git add app/services/synthetic_qa.py tests/test_synthetic_qa.py
git commit -m "feat(synthetic-qa): add SyntheticQAGenerator pipeline class"
```

---

## Task 10: Resume support — track processed chunks via JSONL meta

**Files:**
- Modify: `app/services/synthetic_qa.py`
- Modify: `tests/test_synthetic_qa.py`

When the CLI is restarted, it re-reads the existing output JSONL, extracts already-seen `source_chunk_id` values, and skips those chunks. We expose this helper as a pure function on the module.

- [ ] **Step 10.1: Write failing tests**

Append to `tests/test_synthetic_qa.py`:

```python
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
```

- [ ] **Step 10.2: Run tests to verify failure**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v -k load_processed`
Expected: 4 tests FAIL with `ImportError`

- [ ] **Step 10.3: Implement load_processed_chunk_ids**

Add to `app/services/synthetic_qa.py` (after the generator class):

```python
from pathlib import Path


def load_processed_chunk_ids(path: Path) -> set[int]:
    """Inspect *path* (an existing JSONL) and return chunk ids seen so far.

    Lines without a ``meta.source_chunk_id`` are silently ignored, as
    are lines that fail to parse — the CLI logs them but never aborts
    a resume operation on a malformed entry.
    """

    processed: set[int] = set()
    if not path.exists():
        return processed

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.debug("Skipping malformed JSONL line during resume scan")
                continue
            meta = data.get("meta") if isinstance(data, dict) else None
            if not isinstance(meta, dict):
                continue
            raw_id = meta.get("source_chunk_id")
            try:
                processed.add(int(raw_id))
            except (TypeError, ValueError):
                continue
    return processed
```

Update `__all__` to add `load_processed_chunk_ids`.

- [ ] **Step 10.4: Run tests to verify pass**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v`
Expected: All 43 tests PASS

- [ ] **Step 10.5: Commit**

```bash
git add app/services/synthetic_qa.py tests/test_synthetic_qa.py
git commit -m "feat(synthetic-qa): add resume-from-JSONL helper"
```

---

## Task 11: Validate against `scripts/validate_dataset.py`

**Files:**
- Modify: `tests/test_synthetic_qa.py`

End-to-end sanity check: a JSONL produced by `QAPair.to_jsonl_line()` round-trips cleanly through `validate_dataset.load_examples()` (the parsing entry point used by the validation CLI).

- [ ] **Step 11.1: Write failing test**

Append to `tests/test_synthetic_qa.py`:

```python
def test_jsonl_output_is_consumed_by_validate_dataset(tmp_path):
    """End-to-end: QAPair JSONL must parse via validate_dataset.load_examples."""
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
```

- [ ] **Step 11.2: Run test**

Run: `py -3 -m pytest tests/test_synthetic_qa.py::test_jsonl_output_is_consumed_by_validate_dataset -v`
Expected: PASS (because QAPair fields already match `validate_dataset`'s expected schema)

If the test FAILS — that means the output schema diverged. Re-check Task 2's `to_dict()` against the field lists `PROMPT_FIELDS` / `INPUT_FIELDS` / `RESPONSE_FIELDS` in `scripts/validate_dataset.py`. Fix `to_dict()` until the test passes.

- [ ] **Step 11.3: Commit**

```bash
git add tests/test_synthetic_qa.py
git commit -m "test(synthetic-qa): assert JSONL is consumable by validate_dataset"
```

---

## Task 12: KB chunk iterator

**Files:**
- Modify: `app/services/synthetic_qa.py`
- Modify: `tests/test_synthetic_qa.py`

We need a helper to list chunks from a `KnowledgeBaseStore` for the CLI to feed into the generator. `KnowledgeBaseStore` already has `list_documents()` but not a chunk-level iterator. We add a thin pure helper that does the SQL query and yields `(chunk_id, chunk_text)` tuples. Keeping the SQL inside `synthetic_qa.py` keeps `KnowledgeBaseStore` untouched.

- [ ] **Step 12.1: Write failing test**

Append to `tests/test_synthetic_qa.py`:

```python
def test_iter_chunks_yields_all_rows_from_store(tmp_path):
    """iter_chunks reads chunks directly via a KnowledgeBaseStore."""
    from app.services.kb_store import KnowledgeBaseStore
    from app.services.synthetic_qa import iter_chunks

    db_path = tmp_path / "kb.sqlite"
    store = KnowledgeBaseStore(db_path=db_path)
    store.add_document(
        title="Regulation A",
        text="Section one talks about safety procedures. " * 20
        + "Section two covers reporting. " * 20,
    )

    chunks = list(iter_chunks(store))

    assert len(chunks) >= 2
    for chunk_id, chunk_text in chunks:
        assert isinstance(chunk_id, int)
        assert isinstance(chunk_text, str)
        assert chunk_text.strip()


def test_iter_chunks_filter_by_document_id(tmp_path):
    from app.services.kb_store import KnowledgeBaseStore
    from app.services.synthetic_qa import iter_chunks

    db_path = tmp_path / "kb.sqlite"
    store = KnowledgeBaseStore(db_path=db_path)
    doc_a = store.add_document(title="A", text="alpha " * 200)
    store.add_document(title="B", text="beta " * 200)

    chunks_a = list(iter_chunks(store, document_id=doc_a.id))

    for chunk_id, chunk_text in chunks_a:
        assert "alpha" in chunk_text
```

- [ ] **Step 12.2: Run tests to verify failure**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v -k iter_chunks`
Expected: 2 tests FAIL with `ImportError`

- [ ] **Step 12.3: Implement iter_chunks**

Add to `app/services/synthetic_qa.py` (near the bottom, before `__all__`):

```python
from typing import Iterator


def iter_chunks(store, *, document_id: int | None = None) -> Iterator[tuple[int, str]]:
    """Yield ``(chunk_id, text)`` for every chunk stored in *store*.

    ``store`` is a :class:`KnowledgeBaseStore` instance. We bypass the
    higher-level ``search``/``list_documents`` APIs because we want
    every chunk verbatim, not ranked or paginated. ``document_id``
    restricts iteration to one document when set.
    """

    sql = "SELECT id, text FROM kb_chunks"
    params: tuple = ()
    if document_id is not None:
        sql += " WHERE document_id = ?"
        params = (int(document_id),)
    sql += " ORDER BY id ASC"

    # KnowledgeBaseStore exposes a private _connect helper used by all
    # its query methods. We reuse it instead of opening a new sqlite3
    # connection so locking and pragmas stay consistent.
    with store._connect() as conn:  # noqa: SLF001 — intentional reuse of internal connection
        for row in conn.execute(sql, params):
            chunk_id = int(row[0])
            text = str(row[1] or "").strip()
            if not text:
                continue
            yield chunk_id, text
```

Update `__all__` to add `iter_chunks`.

- [ ] **Step 12.4: Run tests to verify pass**

Run: `py -3 -m pytest tests/test_synthetic_qa.py -v`
Expected: All 45 tests PASS

- [ ] **Step 12.5: Commit**

```bash
git add app/services/synthetic_qa.py tests/test_synthetic_qa.py
git commit -m "feat(synthetic-qa): add KB chunk iterator helper"
```

---

## Task 13: CLI script — argparse, wiring, JSONL writer, resume

**Files:**
- Create: `scripts/generate_synthetic_qa.py`
- Create: `tests/scripts/test_generate_synthetic_qa.py`

The script binds CLI arguments to module helpers and writes the JSONL incrementally so a SIGINT preserves progress.

- [ ] **Step 13.1: Write failing CLI smoke test**

Create `tests/scripts/test_generate_synthetic_qa.py`:

```python
"""CLI smoke tests for scripts/generate_synthetic_qa.py."""

from __future__ import annotations

import json
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
        return "fake"

    @property
    def model(self) -> str:
        return "fake-model"

    def generate(self, prompt, *, system=None, max_tokens=None, temperature=None):
        self.calls += 1
        text = self._responses[(self.calls - 1) % len(self._responses)]
        return LLMResponse(text=text, provider="fake", model="fake-model", elapsed_ms=1.0)


@pytest.fixture()
def populated_store(tmp_path: Path) -> KnowledgeBaseStore:
    store = KnowledgeBaseStore(db_path=tmp_path / "kb.sqlite")
    store.add_document(
        title="Reg",
        text="The annual leave is twenty-eight days. " * 30,
    )
    return store


def test_cli_writes_jsonl_consumable_by_validate_dataset(tmp_path, populated_store, monkeypatch):
    from scripts import generate_synthetic_qa as cli
    from scripts import validate_dataset as vd

    # Stable response so length and self-consistency filters pass
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
    exit_code = cli.main([
        "--corpus", str(tmp_path / "kb.sqlite"),
        "--provider", "deepseek",
        "--mode", "single",
        "--output", str(out_path),
        "--max-budget-usd", "0.0000001",  # absurdly low
    ])

    assert exit_code != 0
    assert not out_path.exists() or out_path.read_text(encoding="utf-8") == ""
```

- [ ] **Step 13.2: Run tests to verify failure**

Run: `py -3 -m pytest tests/scripts/test_generate_synthetic_qa.py -v`
Expected: All FAIL with `ModuleNotFoundError: No module named 'scripts.generate_synthetic_qa'`

- [ ] **Step 13.3: Implement CLI script**

Create `scripts/generate_synthetic_qa.py`:

```python
#!/usr/bin/env python3
"""Generate a synthetic Q&A dataset from a KB corpus.

This is the CLI wrapper for Workstream 1 of the Pack B++ ML
strengthening plan. The pure logic lives in
``app.services.synthetic_qa``; this module only handles argument
parsing, provider/store wiring, the streaming JSONL writer, the
budget guard and resume support.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from app.services.kb_llm import (
    LLMUnavailable,
    OpenAICompatibleProvider,
    build_provider,
    select_provider,
)
from app.services.kb_store import KnowledgeBaseStore
from app.services.synthetic_qa import (
    GenerationMode,
    QAPair,
    SyntheticQAGenerator,
    estimate_total_cost_usd,
    iter_chunks,
    load_processed_chunk_ids,
)

LOGGER = logging.getLogger("scripts.generate_synthetic_qa")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Q&A dataset from a KB corpus via a teacher LLM."
    )
    parser.add_argument(
        "--corpus", required=True, type=Path,
        help="Path to KB SQLite file (e.g. var/data/kb_mvp.sqlite).",
    )
    parser.add_argument(
        "--provider", default=None,
        help="Teacher LLM provider name (deepseek, groq, openrouter, openai, ollama, custom). "
             "Defaults to KB_LLM_PROVIDER env or auto-selection.",
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in GenerationMode],
        default=GenerationMode.SINGLE.value,
        help="Generation strategy (default: single).",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output JSONL file (created or appended to).",
    )
    parser.add_argument(
        "--document-id", type=int, default=None,
        help="Restrict generation to one document id (default: all chunks).",
    )
    parser.add_argument(
        "--multi-hop-chunks", type=int, default=3,
        help="How many chunks to combine when mode=multi-hop (default 3).",
    )
    parser.add_argument(
        "--max-budget-usd", type=float, default=5.0,
        help="Abort if estimated cost exceeds this many USD (default 5.0).",
    )
    parser.add_argument(
        "--no-budget-guard", action="store_true",
        help="Disable the budget guard entirely (use with care).",
    )
    parser.add_argument(
        "--no-self-consistency", action="store_true",
        help="Disable the second-generation self-consistency check.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip chunks already represented in the output JSONL.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s %(name)s: %(message)s",
    )


def _load_store(args: argparse.Namespace) -> KnowledgeBaseStore:
    """Open the KnowledgeBaseStore pointed at by --corpus."""

    if not args.corpus.is_file():
        raise SystemExit(f"Corpus file not found: {args.corpus}")
    return KnowledgeBaseStore(db_path=args.corpus)


def _load_provider(args: argparse.Namespace) -> OpenAICompatibleProvider:
    """Build the teacher provider from --provider or env autoselection."""

    if args.provider:
        try:
            return build_provider(args.provider)
        except LLMUnavailable as exc:
            raise SystemExit(f"LLM provider unusable: {exc}")
    selected = select_provider()
    if selected is None:
        raise SystemExit(
            "No LLM provider configured. Set KB_LLM_PROVIDER or one of "
            "DEEPSEEK_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY."
        )
    return selected


def _enforce_budget(
    args: argparse.Namespace,
    provider: OpenAICompatibleProvider,
    chunk_chars: list[int],
) -> None:
    if args.no_budget_guard:
        LOGGER.info("Budget guard disabled by --no-budget-guard")
        return

    mode = GenerationMode(args.mode)
    estimate = estimate_total_cost_usd(
        provider=provider.name,
        model=provider.model,
        mode=mode,
        chunk_chars=chunk_chars,
    )
    if estimate is None:
        LOGGER.warning(
            "No pricing data for (%s, %s); budget guard disabled.",
            provider.name, provider.model,
        )
        return

    LOGGER.info("Estimated cost: $%.4f (budget cap $%.4f)", estimate, args.max_budget_usd)
    if estimate > args.max_budget_usd:
        raise SystemExit(
            f"Estimated cost ${estimate:.4f} exceeds budget cap ${args.max_budget_usd:.4f}. "
            "Increase --max-budget-usd or trim the corpus."
        )


def _select_chunk_batches(
    chunks: list[tuple[int, str]],
    mode: GenerationMode,
    multi_hop_size: int,
) -> list[list[tuple[int, str]]]:
    if mode is not GenerationMode.MULTI_HOP:
        return [[chunk] for chunk in chunks]
    if multi_hop_size < 2:
        raise SystemExit("--multi-hop-chunks must be >= 2 for multi-hop mode")
    return [
        chunks[i : i + multi_hop_size]
        for i in range(0, len(chunks), multi_hop_size)
        if len(chunks[i : i + multi_hop_size]) >= 2
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.log_level)

    store = _load_store(args)
    provider = _load_provider(args)
    mode = GenerationMode(args.mode)

    LOGGER.info(
        "Generator config: provider=%s model=%s mode=%s",
        provider.name, provider.model, mode.value,
    )

    all_chunks = list(iter_chunks(store, document_id=args.document_id))
    if not all_chunks:
        LOGGER.warning("No chunks found in corpus; nothing to do.")
        return 0

    processed: set[int] = set()
    if args.resume:
        processed = load_processed_chunk_ids(args.output)
        LOGGER.info("Resume: %d chunks already in %s", len(processed), args.output)

    remaining = [(cid, text) for cid, text in all_chunks if cid not in processed]
    if not remaining:
        LOGGER.info("All chunks already processed; exiting cleanly.")
        return 0

    _enforce_budget(args, provider, [len(text) for _, text in remaining])

    generator = SyntheticQAGenerator(
        provider=provider,
        check_self_consistency=not args.no_self_consistency,
    )

    batches = _select_chunk_batches(remaining, mode, args.multi_hop_chunks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if args.resume and args.output.exists() else "w"

    written = 0
    with args.output.open(open_mode, encoding="utf-8") as handle:
        for batch in batches:
            chunk_ids = [cid for cid, _ in batch]
            chunk_texts = [text for _, text in batch]
            try:
                pairs = generator.generate_for_chunk(
                    chunks=chunk_texts,
                    chunk_ids=chunk_ids,
                    mode=mode,
                )
            except Exception:
                LOGGER.exception("Generation failed for chunks %s", chunk_ids)
                continue

            for pair in pairs:
                handle.write(pair.to_jsonl_line())
                handle.flush()
                written += 1

            LOGGER.info(
                "Batch chunks=%s kept=%d total=%d",
                chunk_ids, len(pairs), written,
            )

    LOGGER.info("Done: %d Q&A pairs written to %s", written, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
```

- [ ] **Step 13.4: Run CLI tests to verify pass**

Run: `py -3 -m pytest tests/scripts/test_generate_synthetic_qa.py -v`
Expected: All 3 tests PASS

If the budget-guard test fails because pricing for (`fake`, `fake-model`) returns `None` and the guard is silently disabled, look at the test setup: it uses `--provider deepseek` to land in the pricing table. Pricing must be present for `("deepseek", "fake-model")` only if the test asserts the guard fires. Re-check `estimate_chunk_cost_usd` — it should accept the deepseek + fake-model key as unknown and skip; the third test uses `--provider deepseek` but the fake_provider reports `model="fake-model"` so pricing is `None` and the guard wouldn't trigger. Fix: pass `--provider` and have `_load_provider` use it (it already does in the test via monkeypatch). The test needs either (a) a pricing entry for deepseek/fake-model or (b) to set a different combination. **Adjust the third test to use `provider.model = "deepseek-chat"`** — update `_FakeProvider.model` property accordingly:

```python
@property
def model(self) -> str:
    return "deepseek-chat"  # so pricing table lookup succeeds
```

Apply this fix to the test file and re-run.

- [ ] **Step 13.5: Run full test suite for regression**

Run: `py -3 -m pytest tests/test_synthetic_qa.py tests/scripts/test_generate_synthetic_qa.py -v`
Expected: All 48 tests PASS

- [ ] **Step 13.6: Commit**

```bash
git add scripts/generate_synthetic_qa.py tests/scripts/test_generate_synthetic_qa.py
git commit -m "feat(synthetic-qa): add CLI script with resume and budget guard"
```

---

## Task 14: README documentation

**Files:**
- Modify: `README.md`

Add a usage block so an operator can run the script from a clean checkout.

- [ ] **Step 14.1: Read current README "обучение LoRA" section**

Run: `py -3 -c "print(open('README.md', encoding='utf-8').read())" | head -n 50`

Locate the heading `## Как обучить свой LoRA-адаптер` to insert the new block right after it.

- [ ] **Step 14.2: Insert the new section**

Use the Edit tool to add a subsection right under `## Как обучить свой LoRA-адаптер` (before step `1. **Подготовьте датасет.**`):

```markdown
### Автогенерация датасета через teacher-LLM

Вместо ручного составления `data/dev.jsonl` можно сгенерировать обучающие
пары вопрос-ответ автоматически из уже загруженного KB-корпуса:

```bash
# DeepSeek (дешёвый teacher, ~$0.50 за 1000 Q&A)
export DEEPSEEK_API_KEY=sk-...
python -m scripts.generate_synthetic_qa \
    --corpus var/data/kb_mvp.sqlite \
    --provider deepseek \
    --mode single \
    --output data/lora/synthetic.jsonl \
    --max-budget-usd 2.0
```

Поддерживаемые режимы (`--mode`):

| Режим | Описание |
|------|----------|
| `single` | Один Q&A на чанк (быстро, минимально) |
| `paraphrase` | Три перефразирования одного вопроса (аугментация) |
| `multi-hop` | Вопрос, требующий объединения 2-3 чанков (сложнее) |

Полезные флаги:

- `--resume` — продолжить с того места, где остановилась прошлая
  запуск (читает уже записанный JSONL, пропускает обработанные чанки).
- `--document-id N` — генерировать только по одному документу.
- `--no-self-consistency` — отключить проверку повторной генерации
  (быстрее, но качество ниже).
- `--no-budget-guard` — снять ограничение по стоимости (use with care).

Сгенерированный JSONL совместим с `scripts/validate_dataset.py` и
`scripts/train_lora.py` без преобразований:

```bash
python scripts/validate_dataset.py \
    --path data/lora/synthetic.jsonl \
    --base-model meta-llama/Llama-3-8b-Instruct
```
```

(Use the Edit tool: `old_string` = the exact line `1. **Подготовьте датасет.**` together with one preceding line; `new_string` = the new subsection above + the original line.)

- [ ] **Step 14.3: Commit**

```bash
git add README.md
git commit -m "docs(synthetic-qa): document scripts/generate_synthetic_qa.py usage"
```

---

## Task 15: Acceptance smoke run against the spec

**Files:**
- (no code changes)

Verify the spec's quantitative acceptance criteria on the dev machine, before declaring W1 done.

- [ ] **Step 15.1: Prepare a 500-chunk corpus**

If you don't already have a populated KB SQLite, run:

```bash
mkdir -p var/data
python -m scripts.dev_server_mvp &
# Use the running server's /api/kb/documents/upload to push 5-10 PDFs/DOCXes
# until you have ~500 chunks in var/data/kb_mvp.sqlite.
```

Verify chunk count:

```bash
py -3 -c "import sqlite3; print(sqlite3.connect('var/data/kb_mvp.sqlite').execute('SELECT COUNT(*) FROM kb_chunks').fetchone())"
```

Expected: a value close to 500 (within 100 either side is fine).

- [ ] **Step 15.2: Run generation with DeepSeek**

```bash
export DEEPSEEK_API_KEY=sk-...   # must be set
time python -m scripts.generate_synthetic_qa \
    --corpus var/data/kb_mvp.sqlite \
    --provider deepseek \
    --mode single \
    --output data/lora/synthetic_test.jsonl \
    --max-budget-usd 1.0 \
    --log-level INFO
```

Expected:
- Run completes in <30 minutes (spec acceptance: 1000 pairs / 500 chunks ≤ 30 min)
- Output file contains ~500-2000 lines depending on filter survival rate

- [ ] **Step 15.3: Validate the output**

```bash
python scripts/validate_dataset.py \
    --path data/lora/synthetic_test.jsonl \
    --base-model sshleifer/tiny-gpt2 \
    --max-seq-len 1024
```

Expected: validation reports `dataset_size` ≈ 500-2000, ≥95% pairs pass without warnings (spec acceptance).

Inspect the human-readable report at `data/lora/synthetic_test.md`:

```bash
cat data/lora/synthetic_test.md
```

- [ ] **Step 15.4: Final commit (no code, just a marker)**

If any tweaks to thresholds were applied above, commit them. Otherwise:

```bash
echo "W1 acceptance smoke passed on $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> docs/superpowers/plans/2026-05-25-w1-synthetic-qa-generation-plan.md
git add docs/superpowers/plans/2026-05-25-w1-synthetic-qa-generation-plan.md
git commit -m "docs(w1): record W1 acceptance smoke run timestamp"
```

---

## Done definition

W1 is complete when all of the following are true:

1. All 48 automated tests (45 in `tests/test_synthetic_qa.py` + 3 in `tests/scripts/test_generate_synthetic_qa.py`) pass on a clean checkout.
2. The CLI smoke run (Task 15) produces a JSONL that `scripts/validate_dataset.py` accepts with ≥95% pass rate.
3. README documents the script.
4. All changes are committed to the working branch (one task per commit, atomic).

After W1 ships, the next workstream candidates per the spec dependency graph are W2 (architecture-aware training) and W5 (RAGAS evaluation) — they can be planned in parallel.
