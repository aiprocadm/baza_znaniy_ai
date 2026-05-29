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
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Iterator, Mapping, Sequence, TypeVar

from app.services.synthetic_qa import QAPair

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

_EnumT = TypeVar("_EnumT", bound=Enum)


def apportion_counts(
    proportions: Mapping[_EnumT, float],
    *,
    total: int,
) -> dict[_EnumT, int]:
    """Hamilton's largest-remainder method.

    Given target shares summing to 1.0, return integer counts per
    enum member whose sum equals ``total`` exactly. Deterministic
    ordering — driven by the order of keys in ``proportions`` —
    breaks remainder ties.

    Generic over any :class:`enum.Enum` subclass so both W3
    (RAGVariant) and W4 (RejectStrategy) can share this helper.
    """

    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    share_sum = sum(proportions.values())
    if abs(share_sum - 1.0) > _PROPORTION_TOLERANCE:
        raise ValueError(
            f"proportions must sum to 1.0 (within {_PROPORTION_TOLERANCE}); got {share_sum}"
        )

    ordered = list(proportions.keys())
    counts: dict[_EnumT, int] = {member: 0 for member in ordered}
    if total == 0:
        return counts

    raw = [(member, proportions[member] * total) for member in ordered]
    floors = [(member, int(value)) for member, value in raw]
    assigned = sum(c for _, c in floors)
    leftover = total - assigned

    remainders = sorted(
        ((member, value - int(value)) for member, value in raw),
        key=lambda item: (-item[1], ordered.index(item[0])),
    )
    for member, count in floors:
        counts[member] = count
    for i in range(leftover):
        counts[remainders[i][0]] += 1
    return counts


# Retriever callbacks receive (query, top_k) and return any sequence
# of objects exposing ``.chunk_index`` (int) and ``.text`` (str). That
# is intentionally a subset of ``kb_store.SearchHit`` so unit tests can
# pass lightweight fakes without importing the heavy real type.
Retriever = Callable[[str, int], Sequence[object]]


def _join_chunks(hits: Sequence[object]) -> str:
    blocks: list[str] = []
    for hit in hits:
        cid = int(getattr(hit, "chunk_index"))
        text = str(getattr(hit, "text", "")).strip()
        if text:
            blocks.append(f"Фрагмент [doc_chunk:{cid}]:\n{text}")
    return "\n\n".join(blocks)


def _chunk_ids(hits: Sequence[object]) -> tuple[int, ...]:
    return tuple(int(getattr(hit, "chunk_index")) for hit in hits)


def build_relevant_sample(
    seed: QAPair,
    *,
    retriever: Retriever,
    top_k: int = 3,
) -> RAGSample | None:
    """Promote a seed Q&A to a RELEVANT variant by attaching retrieved context.

    Returns ``None`` when the seed chunk is not in the top-k retrieval
    set — that means the seed answer cannot be grounded in the context
    we plan to feed at inference time, so training on it would teach
    the model to hallucinate.
    """

    hits = list(retriever(seed.instruction, top_k))
    ids = _chunk_ids(hits)
    if seed.source_chunk_id not in ids:
        return None

    return RAGSample(
        instruction=seed.instruction,
        input=seed.input,
        output=seed.output,
        retrieved_context=_join_chunks(hits),
        variant=RAGVariant.RELEVANT,
        source_chunk_id=seed.source_chunk_id,
        retrieved_chunk_ids=ids,
    )


IRRELEVANT_REFUSAL = "Не удалось найти в документах информацию для ответа."


def build_irrelevant_sample(
    seed: QAPair,
    *,
    negative_chunks: Sequence[object],
) -> RAGSample:
    """Pair the seed question with unrelated context and a refusal answer.

    Caller is responsible for picking truly unrelated ``negative_chunks``
    (e.g. from a different document). The W3 spec target share is 15 %.
    """

    return RAGSample(
        instruction=seed.instruction,
        input=seed.input,
        output=IRRELEVANT_REFUSAL,
        retrieved_context=_join_chunks(negative_chunks),
        variant=RAGVariant.IRRELEVANT,
        source_chunk_id=seed.source_chunk_id,
        retrieved_chunk_ids=_chunk_ids(negative_chunks),
    )


PARTIAL_PREFIX = "По доступным фрагментам: "


def build_partial_sample(
    seed: QAPair,
    *,
    seed_hit: object,
    distractor_chunks: Sequence[object],
) -> RAGSample:
    """Mix the seed chunk with distractors and hedge the answer.

    The hedged output keeps the seed citation intact so the model
    still learns the citation format; the prefix teaches caution when
    only part of the context is on-topic.
    """

    mixed = [seed_hit, *distractor_chunks]
    hedged_output = PARTIAL_PREFIX + seed.output
    return RAGSample(
        instruction=seed.instruction,
        input=seed.input,
        output=hedged_output,
        retrieved_context=_join_chunks(mixed),
        variant=RAGVariant.PARTIAL,
        source_chunk_id=seed.source_chunk_id,
        retrieved_chunk_ids=_chunk_ids(mixed),
    )


_CITATION_RE = re.compile(r"\s*\[doc_chunk:\d+\]\s*")


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub(" ", text).strip()


def strip_citations(text: str) -> str:
    """Remove ``[doc_chunk:N]`` markers from ``text``.

    Public alias of :func:`_strip_citations`. Used by W4
    (DPO post-training) to construct the ``NO_CITATION`` reject branch.
    """

    return _strip_citations(text)


def build_empty_sample(seed: QAPair) -> RAGSample:
    """Drop retrieved context and citation suffix from a seed Q&A."""

    return RAGSample(
        instruction=seed.instruction,
        input=seed.input,
        output=_strip_citations(seed.output),
        retrieved_context="",
        variant=RAGVariant.EMPTY,
        source_chunk_id=seed.source_chunk_id,
        retrieved_chunk_ids=(),
    )


@dataclass(slots=True)
class RAGSampleBuilder:
    """Orchestrate variant assembly across an iterable of seed Q&A pairs.

    The builder is the only place that knows about proportions; the
    per-variant ``build_*`` helpers stay independent and reusable.
    """

    retriever: Retriever
    negative_pool: Sequence[object]
    distractor_pool: Sequence[object]
    proportions: ProportionSpec = field(default_factory=default_proportions)
    top_k: int = 3

    def build(
        self,
        seeds: Iterable[QAPair],
        *,
        total: int,
    ) -> Iterator[RAGSample]:
        counts = apportion_counts(self.proportions, total=total)
        emitted: dict[RAGVariant, int] = {v: 0 for v in RAGVariant}

        order = (
            RAGVariant.RELEVANT,
            RAGVariant.IRRELEVANT,
            RAGVariant.PARTIAL,
            RAGVariant.EMPTY,
        )

        for seed in seeds:
            if sum(emitted.values()) >= total:
                return
            for variant in order:
                if emitted[variant] >= counts[variant]:
                    continue
                sample = self._build_one(seed, variant)
                if sample is None:
                    continue
                emitted[variant] += 1
                yield sample
                break

    def _build_one(self, seed: QAPair, variant: RAGVariant) -> RAGSample | None:
        if variant is RAGVariant.RELEVANT:
            return build_relevant_sample(seed, retriever=self.retriever, top_k=self.top_k)
        if variant is RAGVariant.IRRELEVANT:
            return build_irrelevant_sample(seed, negative_chunks=self.negative_pool)
        if variant is RAGVariant.PARTIAL:
            hits = list(self.retriever(seed.instruction, self.top_k))
            seed_hit = next(
                (h for h in hits if int(getattr(h, "chunk_index")) == seed.source_chunk_id),
                None,
            )
            if seed_hit is None:
                return None
            return build_partial_sample(
                seed,
                seed_hit=seed_hit,
                distractor_chunks=self.distractor_pool,
            )
        if variant is RAGVariant.EMPTY:
            return build_empty_sample(seed)
        return None
