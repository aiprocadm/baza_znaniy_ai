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
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Iterator, Mapping

from app.services.rag_dataset import apportion_counts, strip_citations
from app.services.synthetic_qa import QAPair

LOGGER = logging.getLogger(__name__)

_CITATION_MARKER = "[doc_chunk:"
_FAKE_CITATION_RE = re.compile(r"\[doc_chunk:(\d+)\]")
_FAKE_CHUNK_RANGE = (900, 999)

TeacherProvider = Callable[[str], str]


_GENERIC_TEACHER_PROMPT = (
    "Ответь на вопрос пользователя, опираясь только на свои общие знания. "
    "НЕ используй никаких документов или цитат. Не указывай источников.\n\n"
    "Вопрос: {question}"
)


_HALLUCINATION_TEACHER_PROMPT = (
    "Сгенерируй правдоподобный ответ на вопрос и обязательно сошлись на "
    "несуществующий документ в формате [doc_chunk:N] где N >= 900. "
    "Это специальный обучающий пример: модель должна научиться НЕ давать "
    "такие выдуманные ссылки.\n\n"
    "Вопрос: {question}"
)


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


def build_no_citation_pair(seed: QAPair) -> DPOPair | None:
    """Strip the citation suffix from the chosen answer to form ``rejected``.

    Returns ``None`` when the seed has no citation marker — there is
    no signal to learn from in that case.
    """

    if _CITATION_MARKER not in seed.output:
        return None
    rejected = strip_citations(seed.output)
    return DPOPair(
        prompt=seed.instruction,
        chosen=seed.output,
        rejected=rejected,
        strategy=RejectStrategy.NO_CITATION,
        source="synthetic",
        source_chunk_id=seed.source_chunk_id,
        feedback_ids=(),
    )


def _ask_teacher_generic(question: str, teacher: TeacherProvider) -> str:
    return teacher(_GENERIC_TEACHER_PROMPT.format(question=question))


def build_generic_pair(
    seed: QAPair,
    *,
    teacher: TeacherProvider,
) -> DPOPair | None:
    """Ask the teacher to answer **without** the retrieved chunk.

    Returns ``None`` when the teacher response is empty or whitespace —
    that means the call failed silently and the pair would teach noise.
    """

    raw = _ask_teacher_generic(seed.instruction, teacher).strip()
    if not raw:
        return None
    # Defensive: if the teacher leaked a citation marker, strip it so
    # the rejected branch stays cleanly ungrounded.
    rejected = strip_citations(raw)
    return DPOPair(
        prompt=seed.instruction,
        chosen=seed.output,
        rejected=rejected,
        strategy=RejectStrategy.GENERIC,
        source="synthetic",
        source_chunk_id=seed.source_chunk_id,
        feedback_ids=(),
    )


def build_hallucination_pair(
    seed: QAPair,
    *,
    teacher: TeacherProvider,
    rng: random.Random | None = None,
) -> DPOPair | None:
    """Ask the teacher for an answer with an **invented** ``[doc_chunk:9XX]``.

    Coerces the citation into the 900-999 fake range if the teacher
    cooperated but used a different id; appends a fresh one if the
    teacher returned no marker at all. Returns ``None`` on empty
    response.
    """

    raw = teacher(_HALLUCINATION_TEACHER_PROMPT.format(question=seed.instruction)).strip()
    if not raw:
        return None

    rng = rng or random.Random(seed.source_chunk_id)
    fake_id = rng.randint(*_FAKE_CHUNK_RANGE)

    match = _FAKE_CITATION_RE.search(raw)
    if match is None:
        rejected = f"{raw} [doc_chunk:{fake_id}]"
    else:
        existing = int(match.group(1))
        if _FAKE_CHUNK_RANGE[0] <= existing <= _FAKE_CHUNK_RANGE[1]:
            rejected = raw
        else:
            rejected = _FAKE_CITATION_RE.sub(f"[doc_chunk:{fake_id}]", raw, count=1)

    return DPOPair(
        prompt=seed.instruction,
        chosen=seed.output,
        rejected=rejected,
        strategy=RejectStrategy.HALLUCINATION,
        source="synthetic",
        source_chunk_id=seed.source_chunk_id,
        feedback_ids=(),
    )


SyntheticProportions = Mapping[RejectStrategy, float]


def default_synthetic_proportions() -> dict[RejectStrategy, float]:
    """Spec defaults: 40 % NO_CITATION (free) / 30 % GENERIC / 30 % HALLUCINATION."""

    return {
        RejectStrategy.NO_CITATION: 0.40,
        RejectStrategy.GENERIC: 0.30,
        RejectStrategy.HALLUCINATION: 0.30,
    }


@dataclass(slots=True)
class DPOPairBuilder:
    """Orchestrate preference-pair assembly across synthetic strategies."""

    teacher: TeacherProvider
    proportions: SyntheticProportions = field(default_factory=default_synthetic_proportions)

    def build(
        self,
        seeds: Iterable[QAPair],
        *,
        total: int,
    ) -> Iterator[DPOPair]:
        counts = apportion_counts(self.proportions, total=total)
        emitted: dict[RejectStrategy, int] = {s: 0 for s in self.proportions}

        priority = (
            RejectStrategy.NO_CITATION,
            RejectStrategy.GENERIC,
            RejectStrategy.HALLUCINATION,
        )

        for seed in seeds:
            if sum(emitted.values()) >= total:
                return
            for strategy in priority:
                if emitted[strategy] >= counts.get(strategy, 0):
                    continue
                pair = self._build_one(seed, strategy)
                if pair is None:
                    continue
                emitted[strategy] += 1
                yield pair
                break

    def _build_one(self, seed: QAPair, strategy: RejectStrategy) -> DPOPair | None:
        if strategy is RejectStrategy.NO_CITATION:
            return build_no_citation_pair(seed)
        if strategy is RejectStrategy.GENERIC:
            return build_generic_pair(seed, teacher=self.teacher)
        if strategy is RejectStrategy.HALLUCINATION:
            return build_hallucination_pair(seed, teacher=self.teacher)
        return None
