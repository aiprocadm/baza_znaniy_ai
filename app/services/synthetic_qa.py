"""Generate synthetic supervised Q&A datasets from a KB corpus."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Protocol

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


_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    match = _FENCE_PATTERN.match(text)
    return match.group(1) if match else text


def _extract_first_json_payload(text: str) -> str | None:
    """Return the first top-level JSON object or array substring in *text*.

    The scan is string-aware: characters inside JSON string literals are
    ignored so that brackets appearing in an ``output`` value (e.g.
    ``[doc_chunk:3]``) do not derail the matcher.
    """

    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        depth = 0
        start = -1
        in_string = False
        escape = False
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
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
    "load_processed_chunk_ids",
    "iter_chunks",
    "MIN_INSTRUCTION_CHARS",
    "MAX_INSTRUCTION_CHARS",
    "MIN_OUTPUT_CHARS",
    "MAX_OUTPUT_CHARS",
    "DEFAULT_CONSISTENCY_THRESHOLD",
    "CHARS_PER_TOKEN_HEURISTIC",
]
