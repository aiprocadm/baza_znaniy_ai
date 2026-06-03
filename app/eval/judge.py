"""LLM-as-judge: prompt construction + robust verdict parsing.

Parsing mirrors the tolerance of ``synthetic_qa.parse_qa_response`` (markdown
fences, surrounding prose). Scores are 1–5, normalized to [0,1] for aggregation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

JUDGE_SYSTEM = (
    "Ты — строгий оценщик ответов RAG-системы по корпоративным документам. "
    "Оцениваешь ответ только по предоставленному контексту. Возвращаешь строго JSON."
)

SCORE_KEYS: tuple[str, ...] = ("faithfulness", "relevance", "completeness", "citation")


def build_judge_prompt(*, question: str, answer: str, context: str, reference: str = "") -> str:
    ref_block = f"\nЭталонный ответ (для оценки полноты):\n{reference}\n" if reference else ""
    return (
        "Оцени ответ системы по шкале 1–5 по каждому критерию:\n"
        "- faithfulness: каждое утверждение подтверждается контекстом, нет выдумок;\n"
        "- relevance: ответ по существу вопроса;\n"
        "- completeness: ответ полон относительно эталона (если он дан);\n"
        "- citation: ссылки вида [N] соответствуют использованным фрагментам.\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Контекст (фрагменты):\n{context}\n{ref_block}\n"
        f"Ответ системы:\n{answer}\n\n"
        "Верни строго JSON без пояснений: "
        '{"faithfulness":N,"relevance":N,"completeness":N,"citation":N,"rationale":"кратко"}'
    )


@dataclass(frozen=True, slots=True)
class Verdict:
    faithfulness: int
    relevance: int
    completeness: int
    citation: int
    rationale: str = ""

    def normalized(self) -> dict[str, float]:
        return {k: (getattr(self, k) - 1) / 4.0 for k in SCORE_KEYS}


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _clamp(value: object) -> int:
    try:
        n = int(round(float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, n))


def parse_verdict(raw: str) -> Verdict | None:
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return Verdict(
        faithfulness=_clamp(data.get("faithfulness")),
        relevance=_clamp(data.get("relevance")),
        completeness=_clamp(data.get("completeness")),
        citation=_clamp(data.get("citation")),
        rationale=str(data.get("rationale", "")),
    )
