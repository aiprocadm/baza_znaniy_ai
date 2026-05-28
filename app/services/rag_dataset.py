"""Compose a RAG-aware SFT dataset from a KB corpus + teacher LLM.

This module is the pure-logic core of Workstream 3 (RAG-aware
fine-tuning) in the Pack B++ ML strengthening plan. It builds on
W1's :mod:`app.services.synthetic_qa` for seed Q&A generation and
on :class:`app.services.kb_store.KnowledgeBaseStore` for retrieval.

The module is intentionally I/O free: provider, retriever, and chunk
source are injected so the logic is deterministic in tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

LOGGER = logging.getLogger(__name__)


class RAGVariant(str, Enum):
    """The four training-distribution variants from the W3 spec.

    See ``docs/superpowers/specs/2026-05-25-ml-strengthening-pack-b-design.md``
    section "Workstream 3" for the rationale and target proportions.
    """

    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    PARTIAL = "partial"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class RAGSample:
    """One RAG-aware training example ready for SFT.

    The top-level layout (``instruction`` / ``input`` / ``output``) keeps
    ``scripts/validate_dataset.py`` happy. ``retrieved_context`` is the
    new field consumed by ``train_lora.py --prompt-mode rag``. The
    ``meta`` sidecar carries variant + retrieval lineage so resume and
    audit queries work.
    """

    instruction: str
    input: str
    output: str
    retrieved_context: str
    variant: RAGVariant
    source_chunk_id: int
    retrieved_chunk_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "retrieved_context": self.retrieved_context,
            "meta": {
                "source_chunk_id": int(self.source_chunk_id),
                "variant": self.variant.value,
                "retrieved_chunk_ids": [int(c) for c in self.retrieved_chunk_ids],
            },
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"


ProportionSpec = Mapping["RAGVariant", float]


def default_proportions() -> dict[RAGVariant, float]:
    """Return the W3 spec defaults: 70 / 15 / 10 / 5."""

    return {
        RAGVariant.RELEVANT: 0.70,
        RAGVariant.IRRELEVANT: 0.15,
        RAGVariant.PARTIAL: 0.10,
        RAGVariant.EMPTY: 0.05,
    }


_PROPORTION_TOLERANCE = 1e-6


def apportion_counts(
    proportions: ProportionSpec,
    *,
    total: int,
) -> dict[RAGVariant, int]:
    """Hamilton's largest-remainder method.

    Given target shares summing to 1.0, return integer counts per
    variant whose sum equals ``total`` exactly. Deterministic ordering
    (RELEVANT, IRRELEVANT, PARTIAL, EMPTY) breaks remainder ties.
    """

    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    share_sum = sum(proportions.values())
    if abs(share_sum - 1.0) > _PROPORTION_TOLERANCE:
        raise ValueError(
            f"proportions must sum to 1.0 (within {_PROPORTION_TOLERANCE}); got {share_sum}"
        )

    counts: dict[RAGVariant, int] = {v: 0 for v in RAGVariant}
    if total == 0:
        return counts

    raw = [(v, proportions.get(v, 0.0) * total) for v in RAGVariant]
    floors = [(v, int(value)) for v, value in raw]
    assigned = sum(c for _, c in floors)
    leftover = total - assigned

    remainders = sorted(
        ((v, value - int(value)) for v, value in raw),
        key=lambda item: (-item[1], list(RAGVariant).index(item[0])),
    )
    for v, count in floors:
        counts[v] = count
    for i in range(leftover):
        counts[remainders[i][0]] += 1
    return counts
