"""Generate synthetic supervised Q&A datasets from a KB corpus."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum

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
