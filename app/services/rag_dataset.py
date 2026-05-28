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
