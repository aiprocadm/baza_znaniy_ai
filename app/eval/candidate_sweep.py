"""Чистая логика свипа «число кандидатов reranker'а ↔ качество» (без модели, без I/O).

Реконструирует ранжирование «реранк только top-k кандидатов» из уже захваченного
top-N шорт-листа би-энкодера и teacher-скор (bge) на кандидата, поэтому весь свип по
числу кандидатов исполняется детерминированно, не загружая модель. Зеркалит
прод-семантику ``KB_RERANK_CANDIDATES`` (``app/services/kb_rerank.py``: би-энкодер
достаёт ровно ``candidates`` хитов, cross-encoder скорит их все) и тай-брейк реранка
(``app/retriever/rerank.py``: sort по ``(score, -index)`` убыв.).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.eval.metrics import RETRIEVAL_KS, aggregate, score_item


def base_topk(shortlist_keys: Sequence[str], k: int) -> list[str]:
    """Ранжирование без реранка: первые ``k`` ключей в порядке би-энкодера."""
    return list(shortlist_keys[: max(0, k)])


def rerank_topk(
    shortlist_keys: Sequence[str],
    teacher_scores: Sequence[float],
    k: int,
) -> list[str]:
    """Ранжирование «реранк top-k»: ``shortlist_keys[:k]``, отсортированный по
    соответствующему teacher-скору убыв., тай-брейк по исходной позиции.

    Зеркалит ``rerank.py``: ``sort(key=(score, -index), reverse=True)`` — при равных
    скорах меньший исходный индекс идёт раньше. Кандидаты за позицией ``k`` в прод
    вообще не достаются (би-энкодер тянет ровно ``candidates``), поэтому они не
    участвуют.
    """
    cut = max(0, k)
    keys = list(shortlist_keys[:cut])
    scores = list(teacher_scores[:cut])
    order = sorted(range(len(keys)), key=lambda i: (scores[i], -i), reverse=True)
    return [keys[i] for i in order]


def sweep_quality(
    items: Sequence[Mapping[str, Any]],
    candidate_ks: Sequence[int],
    metric_ks: Sequence[int] = RETRIEVAL_KS,
) -> dict[int, dict[str, dict[str, float]]]:
    """На каждое число кандидатов из ``candidate_ks`` — агрегированные метрики base и teacher.

    ``item`` содержит: ``relevant`` (релевантные ключи), ``shortlist_keys`` (top-N
    порядок би-энкодера), ``teacher_scores`` (скор bge на кандидата, той же длины и
    порядка, что ``shortlist_keys``). Возврат: ``{k: {"base": {...}, "teacher": {...}}}``,
    где значения — усреднённые по вопросам метрики из ``app.eval.metrics``.
    """
    out: dict[int, dict[str, dict[str, float]]] = {}
    for k in candidate_ks:
        base_rows: list[dict[str, float]] = []
        teacher_rows: list[dict[str, float]] = []
        for it in items:
            relevant = it["relevant"]
            keys = it["shortlist_keys"]
            scores = it["teacher_scores"]
            base_rows.append(score_item(relevant, base_topk(keys, k), metric_ks))
            teacher_rows.append(score_item(relevant, rerank_topk(keys, scores, k), metric_ks))
        out[k] = {"base": aggregate(base_rows), "teacher": aggregate(teacher_rows)}
    return out


__all__ = ["base_topk", "rerank_topk", "sweep_quality"]
