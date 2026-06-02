"""Score end-to-end answers: deterministic refusal-correctness + LLM-judge.

The answer prompt mirrors the production MVP path (``kb_mvp._build_rag_prompt``
+ ``_RAG_SYSTEM_PROMPT``). ``RAG_SYSTEM_PROMPT`` is pinned equal to the
production constant by a drift test rather than imported at runtime, so this
module stays free of the FastAPI import chain.
"""
from __future__ import annotations

from typing import Sequence

from app.eval.adapter import EvalHit, Retriever
from app.eval.dataset import GoldenItem
from app.eval.judge import JUDGE_SYSTEM, build_judge_prompt, parse_verdict
from app.eval.metrics import aggregate
from app.services.synthetic_qa import LLMProvider, is_refusal

# MUST stay byte-identical to app.api.kb_mvp._RAG_SYSTEM_PROMPT (drift-tested).
RAG_SYSTEM_PROMPT = (
    "Ты — помощник корпоративной базы знаний. Отвечай на русском. "
    "Используй ТОЛЬКО фрагменты из контекста, не выдумывай факты. "
    "Если данных недостаточно — честно сообщи об этом. "
    "В ответе ссылайся на фрагменты в формате [1], [2] там, где они уместны."
)

_CANONICAL_REFUSAL = "не удалось найти"


def looks_like_refusal(text: str) -> bool:
    return is_refusal(text) or _CANONICAL_REFUSAL in (text or "").lower()


def format_context(hits: Sequence[EvalHit]) -> str:
    # Mirrors kb_mvp._format_context: "[i] <source>\n<text>" joined by separators.
    parts = []
    for i, h in enumerate(hits, start=1):
        label = h.title or "фрагмент"
        parts.append(f"[{i}] {label}\n{h.text}")
    return "\n\n---\n\n".join(parts)


def _build_answer_prompt(question: str, context: str) -> str:
    return f"Фрагменты базы знаний:\n{context}\n\nВопрос пользователя: {question}\nОтвет:"


def _generate(provider: LLMProvider, prompt: str, system: str) -> str:
    resp = provider.generate(prompt, system=system)
    return getattr(resp, "text", "") or ""


def evaluate_generation_item(
    item: GoldenItem,
    hits: Sequence[EvalHit],
    *,
    gen_provider: LLMProvider,
    judge_provider: LLMProvider,
) -> dict[str, float]:
    context = format_context(hits)
    answer = _generate(gen_provider, _build_answer_prompt(item.question, context), RAG_SYSTEM_PROMPT)
    if item.expect_refusal:
        return {"refusal_correct": 1.0 if looks_like_refusal(answer) else 0.0}
    jprompt = build_judge_prompt(
        question=item.question, answer=answer, context=context, reference=item.reference_answer
    )
    verdict = parse_verdict(_generate(judge_provider, jprompt, JUDGE_SYSTEM))
    return verdict.normalized() if verdict else {}


def evaluate_generation(
    items: Sequence[GoldenItem],
    retriever: Retriever,
    *,
    gen_provider: LLMProvider,
    judge_provider: LLMProvider,
    top_k: int,
) -> dict[str, object]:
    judge_rows: list[dict[str, float]] = []
    refusal_rows: list[dict[str, float]] = []
    per_item: list[dict[str, object]] = []
    for item in items:
        hits = list(retriever(item.question, top_k))
        row = evaluate_generation_item(
            item, hits, gen_provider=gen_provider, judge_provider=judge_provider
        )
        per_item.append({"question": item.question, "expect_refusal": item.expect_refusal, **row})
        if item.expect_refusal:
            refusal_rows.append(row)
        elif row:
            judge_rows.append(row)

    agg: dict[str, float] = {}
    if judge_rows:
        agg.update(aggregate(judge_rows))
    if refusal_rows:
        agg["refusal_correct"] = aggregate(refusal_rows)["refusal_correct"]

    return {
        "n_answerable": sum(1 for i in items if not i.expect_refusal),
        "n_refusal": sum(1 for i in items if i.expect_refusal),
        "per_item": per_item,
        "aggregate": agg,
    }
