"""Compose a synthetic DPO preference dataset.

Pure-logic core of Workstream 4 (DPO post-training / preference
learning) in the Pack B++ ML strengthening plan. Builds on W1's
:mod:`app.services.synthetic_qa` for seed Q&A and W3's
:mod:`app.services.rag_dataset` (Hamilton apportionment +
citation stripping helpers).

The module is intentionally I/O free: teacher provider and the
seed iterator are injected so the logic is deterministic in tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

LOGGER = logging.getLogger(__name__)


class RejectStrategy(str, Enum):
    """How the ``rejected`` half of a DPO pair was constructed.

    Synthetic branches (`no_citation`, `generic`, `hallucination`) come
    from teacher-LLM or regex generators. Live branches
    (`live_alt`, `live_paired`) come from user feedback collected via
    the ``/api/kb/messages/{id}/feedback`` endpoint.
    """

    NO_CITATION = "no_citation"
    GENERIC = "generic"
    HALLUCINATION = "hallucination"
    LIVE_ALT = "live_alt"
    LIVE_PAIRED = "live_paired"


@dataclass(frozen=True, slots=True)
class DPOPair:
    """One preference pair ready for trl.DPOTrainer.

    Top-level ``prompt / chosen / rejected`` match the trl 0.11
    dataset contract so no transform pass is needed before training.
    """

    prompt: str
    chosen: str
    rejected: str
    strategy: RejectStrategy
    source: str  # "synthetic" or "live"
    source_chunk_id: int | None
    feedback_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "meta": {
                "source": self.source,
                "strategy": self.strategy.value,
                "source_chunk_id": (
                    int(self.source_chunk_id) if self.source_chunk_id is not None else None
                ),
                "feedback_ids": list(self.feedback_ids),
            },
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"
