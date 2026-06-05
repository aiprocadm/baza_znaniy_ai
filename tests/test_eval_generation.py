from dataclasses import dataclass
from app.eval.adapter import EvalHit
from app.eval.dataset import GoldenItem
from app.eval.generation_eval import (
    RAG_SYSTEM_PROMPT,
    looks_like_refusal,
    evaluate_generation,
)


@dataclass
class _Resp:
    text: str


class _Provider:
    """Returns a canned text; records the last prompt it saw."""

    def __init__(self, text):
        self.text = text
        self.last_prompt = None

    name = "fake"
    model = "fake"

    def generate(self, prompt, *, system=None, max_tokens=None, temperature=None):
        self.last_prompt = prompt
        return _Resp(self.text)


def test_system_prompt_matches_production():
    # Drift guard: the eval must answer with the SAME system prompt as the MVP path.
    from app.api.kb_mvp import _RAG_SYSTEM_PROMPT

    assert RAG_SYSTEM_PROMPT == _RAG_SYSTEM_PROMPT


def test_looks_like_refusal():
    assert looks_like_refusal("Не удалось найти в документах информацию для ответа.")
    assert looks_like_refusal("Извините, я не могу ответить.")
    assert not looks_like_refusal("Отпуск — это перерыв [1].")


def test_refusal_item_scored_deterministically():
    items = [GoldenItem("Кто выиграл матч?", (), expect_refusal=True)]

    def retriever(q, k):
        return [EvalHit(1, "нерелевантный текст")]

    gen = _Provider("Не удалось найти в документах информацию для ответа.")
    judge = _Provider("{}")  # must not be consulted for refusal items
    out = evaluate_generation(items, retriever, gen_provider=gen, judge_provider=judge, top_k=5)
    assert out["aggregate"]["refusal_correct"] == 1.0
    assert out["n_refusal"] == 1 and out["n_answerable"] == 0
    assert judge.last_prompt is None


def test_answerable_item_uses_judge():
    items = [GoldenItem("Что такое отпуск?", (7,), reference_answer="перерыв")]

    def retriever(q, k):
        return [EvalHit(7, "Отпуск — перерыв.")]

    gen = _Provider("Отпуск — это перерыв [1].")
    judge = _Provider('{"faithfulness":5,"relevance":5,"completeness":4,"citation":5}')
    out = evaluate_generation(items, retriever, gen_provider=gen, judge_provider=judge, top_k=5)
    assert out["aggregate"]["faithfulness"] == 1.0
    assert out["n_answerable"] == 1
    # The judge saw the generated answer and the retrieved context.
    assert "Отпуск — это перерыв [1]." in judge.last_prompt


def test_prompt_requires_per_claim_citation():
    from app.api.kb_mvp import _RAG_SYSTEM_PROMPT

    assert "[N]" in _RAG_SYSTEM_PROMPT
    assert "Каждое утверждение" in _RAG_SYSTEM_PROMPT


def test_prompt_prescribes_canonical_refusal():
    from app.api.kb_mvp import _RAG_SYSTEM_PROMPT
    from app.services.rag_dataset import IRRELEVANT_REFUSAL

    # The prescribed refusal phrase must be exactly the one the deterministic
    # refusal detector (looks_like_refusal) recognises, so refusal_correct is
    # meaningful once an LLM is configured.
    assert IRRELEVANT_REFUSAL in _RAG_SYSTEM_PROMPT
