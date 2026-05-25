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
