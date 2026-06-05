# Infra-Free RAG Quality Wins — Curated Golden + Prompt Tightening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RAG answer quality measurably improvable *without* standing up a real embedder/LLM: replace the stub eval golden set with a real corpus-pinned one, and tighten the MVP system prompt for per-claim citations + a canonical refusal, with a turnkey runbook for the measurement-gated wins.

**Architecture:** Three independent commits. **PR-G** authors a real `data/eval/golden_curated.jsonl` (+ `.sig.json` sidecar) against the live 48-chunk MVP corpus via a committed builder script, guarded by a new structural test. **PR-E** edits the MVP system prompt constant and its drift-pinned eval copy together, adding two assertions. **PR-R** commits a runbook (and an optional `eval_rag sig` subcommand). Order: PR-G → PR-E (independent; G first). No production behaviour changes beyond the MVP system prompt; no new heavy deps; the two-path API design is untouched.

**Tech Stack:** Python 3 (Windows `py -3` launcher, **no venv**), pytest, ruff, black. Reuses the existing `app/eval/*` harness (`GoldenItem`, `CorpusSignature`, `compute_signature`, `write_signature`, `load_golden`, `read_signature`) and `app/services/kb_store.get_store`.

**Spec:** `docs/superpowers/specs/2026-06-05-rag-curated-golden-and-prompt-tightening-design.md`

---

## Pre-flight (read once, do not skip)

- Windows: every command uses the `py -3` launcher. There is **no** `.venv`; deps are on the user site-packages.
- The live MVP corpus is `var/data/kb_mvp.sqlite` — **1 document / 48 chunks** (occupational-safety outsourcing contract «ПРОМТЕХНОСФЕРА» ↔ «РУСКОНСТРУКТ»). `get_store()` resolves to it when `KB_MVP_DB_PATH` is unset.
- Golden ground-truth ids are **global `kb_chunks.id`** values (range 1–48), the same id `compute_signature` pins as `max_chunk_id`. They are **reindex-stable** (reindex updates chunks in place), so authoring them now against the hashing index is correct.
- Run the whole touched suite at the end: `py -3 -m pytest tests/test_golden_curated.py tests/test_eval_generation.py tests/test_eval_dataset.py -q` (offline, no network).

---

## File Structure

| File | Responsibility | PR |
|---|---|---|
| `scripts/build_curated_golden.py` | **Create.** Committed, re-runnable builder: holds the curated items as readable `GoldenItem`s, emits the JSONL + signature sidecar. The review target. | PR-G |
| `data/eval/golden_curated.jsonl` | **Replace.** Generated artifact (committed) — the real curated golden set. | PR-G |
| `data/eval/golden_curated.sig.json` | **Create.** Generated artifact (committed) — corpus signature sidecar. | PR-G |
| `tests/test_golden_curated.py` | **Create.** Structural guard against re-stubbing (count, refusal probes, chunk-id range, sidecar). | PR-G |
| `app/api/kb_mvp.py` (`_RAG_SYSTEM_PROMPT`, ~L406) | **Modify.** Tightened MVP system prompt. | PR-E |
| `app/eval/generation_eval.py` (`RAG_SYSTEM_PROMPT`, ~L20) | **Modify.** Drift-pinned copy — kept byte-identical. | PR-E |
| `tests/test_eval_generation.py` | **Modify.** Add per-claim-citation + canonical-refusal assertions. | PR-E |
| `docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md` | **Create.** Turnkey "models exist → baseline → queued gates" runbook. | PR-R |
| `scripts/eval_rag.py` (`sig` subcommand) | **Modify (optional).** One-command sidecar refresh. | PR-R |

---

## PR-G — Real curated golden set

### Task G1: Failing structural test

**Files:**
- Test: `tests/test_golden_curated.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_golden_curated.py
from pathlib import Path

from app.eval.dataset import load_golden, read_signature

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "data" / "eval" / "golden_curated.jsonl"
MAX_CHUNK_ID = 48  # kb_mvp corpus snapshot: 1 doc / 48 chunks


def test_curated_golden_is_real_not_stub():
    items = load_golden(GOLDEN)
    assert len(items) >= 18
    assert all("ЗАМЕНИ" not in it.reference_answer for it in items)


def test_curated_golden_has_refusal_probes():
    items = load_golden(GOLDEN)
    refusals = [it for it in items if it.expect_refusal]
    assert len(refusals) >= 3
    assert all(it.relevant_chunk_ids == () for it in refusals)


def test_curated_answerable_items_reference_real_chunks():
    items = load_golden(GOLDEN)
    answerable = [it for it in items if not it.expect_refusal]
    assert len(answerable) >= 12
    for it in answerable:
        assert it.relevant_chunk_ids, it.question
        assert all(1 <= cid <= MAX_CHUNK_ID for cid in it.relevant_chunk_ids), it.question
        assert it.reference_answer.strip(), it.question


def test_curated_golden_has_signature_sidecar():
    sig = read_signature(GOLDEN)
    assert sig is not None
    assert sig.doc_count == 1
    assert sig.max_chunk_id == MAX_CHUNK_ID
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_golden_curated.py -q`
Expected: FAIL — the current stub has 4 items (`<18`), contains `"ЗАМЕНИ"`, only 1 answerable (`<12`), and has no `.sig.json` sidecar.

### Task G2: Curated golden builder

**Files:**
- Create: `scripts/build_curated_golden.py`

- [ ] **Step 1: Write the builder with the real items**

Reference answers below are derived from the contract chunks and are subject to the domain-owner review in Task G3. `relevant_chunk_ids` are global `kb_chunks.id` values verified against the corpus.

```python
# scripts/build_curated_golden.py
"""Build the committed curated golden set for the RAG eval harness.

Re-runnable: regenerates data/eval/golden_curated.jsonl + its corpus-signature
sidecar from the GoldenItem list below. Labels are global kb_chunks.id values
(reindex-stable). Reference answers are derived from the corpus and reviewed by
a domain owner before commit. After a real reindex, re-run to refresh the
sidecar (chunk-ids unchanged; embedder/dim flip).
"""
from __future__ import annotations

from pathlib import Path

from app.eval.adapter import compute_signature
from app.eval.dataset import GoldenItem, save_golden, write_signature
from app.services.kb_store import get_store

GOLDEN = Path("data/eval/golden_curated.jsonl")

# --- Answerable (single-chunk, multi-hop, paraphrase) ---
ANSWERABLE: list[GoldenItem] = [
    GoldenItem("Какова ежемесячная стоимость услуг по договору?", (3,),
               "45 000 рублей в месяц; НДС не облагается (упрощённая система налогообложения у Исполнителя).",
               source="curated"),
    GoldenItem("В какие сроки и на каких условиях Заказчик оплачивает услуги?", (3, 4),
               "Ежемесячно, 100% предоплатой, в течение 5 рабочих дней с момента выставления счёта, но не ранее 5-го числа оплачиваемого месяца.",
               source="curated"),
    GoldenItem("За какой срок Исполнитель разрабатывает первичный пакет документов?", (6, 7),
               "В течение 20 рабочих дней со дня получения полной информации по Брифу.",
               source="curated"),
    GoldenItem("Какая неустойка предусмотрена за просрочку оплаты Заказчиком?", (22,),
               "Пени в размере 0,1% от суммы просроченного платежа за каждый день просрочки.",
               source="curated"),
    GoldenItem("До какой даты действует договор и как он продлевается?", (29,),
               "До 31 декабря 2025 года; продлевается на каждый следующий календарный год, если ни одна сторона не заявит о расторжении за 30 дней до окончания.",
               source="curated"),
    GoldenItem("Каков минимальный период оказания услуг по договору?", (30,),
               "3 месяца с даты подписания; при досрочном расторжении по инициативе Заказчика он оплачивает услуги за полный минимальный период (3 ежемесячных платежа).",
               source="curated"),
    GoldenItem("За сколько дней нужно уведомить о расторжении договора в одностороннем порядке?", (30,),
               "Не менее чем за 30 календарных дней до даты расторжения.",
               source="curated"),
    GoldenItem("Кто выступает Исполнителем и Заказчиком по договору?", (1,),
               "Исполнитель — ООО «ПРОМТЕХНОСФЕРА»; Заказчик — ООО «РУСКОНСТРУКТ Северо-Запад».",
               source="curated"),
    GoldenItem("В каком суде рассматриваются споры по договору?", (28, 29),
               "В Арбитражном суде города Санкт-Петербурга и Ленинградской области, если спор не урегулирован переговорами.",
               source="curated"),
    GoldenItem("Что относится к обстоятельствам непреодолимой силы по договору?", (24,),
               "Пожар, наводнение, землетрясение, забастовки, война и военные действия и иные обстоятельства вне контроля сторон.",
               source="curated"),
    GoldenItem("Включены ли в стоимость услуги по расследованию тяжёлых несчастных случаев?", (38,),
               "Нет. Расследование групповых, тяжёлых или смертельных несчастных случаев не входит в стоимость и оплачивается отдельно.",
               source="curated"),
    GoldenItem("Какие персональные данные сотрудников Заказчик поручает обрабатывать Исполнителю?", (35,),
               "ФИО, дату рождения, должность, наименование структурного подразделения и дату приёма на работу.",
               source="curated"),
    GoldenItem("Что произойдёт, если Заказчик в течение 10 рабочих дней не подпишет Акт и не направит отказ?", (21,),
               "Услуги считаются оказанными надлежащим образом и принятыми в полном объёме.",
               source="curated"),
    GoldenItem("Какие адреса электронной почты признаются для юридически значимой переписки?", (40,),
               "Со стороны Заказчика — snab@rusconstruct.com, со стороны Исполнителя — ot@otsfera.ru.",
               source="curated"),
    # paraphrases (same facts, different wording → retrieval robustness)
    GoldenItem("Сколько в месяц платит Заказчик по этому договору?", (3,),
               "45 000 рублей ежемесячно.",
               source="curated"),
    GoldenItem("Какие пени начисляются при несвоевременной оплате?", (22,),
               "0,1% от просроченной суммы за каждый день просрочки.",
               source="curated"),
]

# --- Refusal probes (no relevant chunk; the system must decline) ---
REFUSALS: list[GoldenItem] = [
    # generic out-of-corpus
    GoldenItem("Какая температура на поверхности Венеры?", (), "", expect_refusal=True, source="curated"),
    GoldenItem("Кто победил в матче вчера вечером?", (), "", expect_refusal=True, source="curated"),
    GoldenItem("Назови рецепт борща из нашей базы знаний.", (), "", expect_refusal=True, source="curated"),
    # plausible-but-out-of-corpus (sound like the contract, but unanswerable from it)
    GoldenItem("Какой размер банковской гарантии предусмотрен договором?", (), "", expect_refusal=True, source="curated"),
    GoldenItem("Какая неустойка предусмотрена для Исполнителя за нарушение сроков разработки документации?", (), "", expect_refusal=True, source="curated"),
]

ITEMS = ANSWERABLE + REFUSALS


def main() -> None:
    save_golden(GOLDEN, ITEMS)
    write_signature(GOLDEN, compute_signature(get_store()))
    print(f"Wrote {len(ITEMS)} curated items ({len(ANSWERABLE)} answerable, "
          f"{len(REFUSALS)} refusal) + signature to {GOLDEN}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the builder to regenerate the golden + sidecar**

Run: `py -3 -m scripts.build_curated_golden`
Expected: `Wrote 21 curated items (16 answerable, 5 refusal) + signature to data/eval/golden_curated.jsonl`
This overwrites `data/eval/golden_curated.jsonl` and creates `data/eval/golden_curated.sig.json` (`{"doc_count": 1, "max_chunk_id": 48, "embedder_name": "hash", "dim": 256}`).

- [ ] **Step 3: Run the test to verify it passes**

Run: `py -3 -m pytest tests/test_golden_curated.py -q`
Expected: PASS (4 tests).

### Task G3: Domain-owner review gate (do not skip)

- [ ] **Step 1: Review the reference answers against the contract**

Ask the user to read the 16 answerable items in `scripts/build_curated_golden.py` (or the emitted `data/eval/golden_curated.jsonl`) and confirm each `reference_answer` is legally accurate for the «ПРОМТЕХНОСФЕРА»↔«РУСКОНСТРУКТ» contract. Apply any wording corrections **in the builder**, then re-run `py -3 -m scripts.build_curated_golden` and `py -3 -m pytest tests/test_golden_curated.py -q`. Per-item escape hatch: an item whose answer the owner does not want to pin may be converted to retrieval-only by emptying its `reference_answer` **and** dropping it from the `>=12 answerable with non-empty reference` count (keep ≥12 fully-specified).

### Task G4: Format and commit

- [ ] **Step 1: Format**

Run: `py -3 -m ruff check scripts/build_curated_golden.py tests/test_golden_curated.py --fix; py -3 -m black scripts/build_curated_golden.py tests/test_golden_curated.py`
Expected: no remaining lint errors.

- [ ] **Step 2: Verify the golden subset is green**

Run: `py -3 -m pytest tests/test_golden_curated.py tests/test_eval_dataset.py -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_curated_golden.py data/eval/golden_curated.jsonl data/eval/golden_curated.sig.json tests/test_golden_curated.py
git commit -m "feat(eval): real curated golden set against the live MVP corpus

Replace the stub golden_curated.jsonl (placeholder answer + unaligned ids)
with 21 reviewed items (16 answerable incl. multi-hop/paraphrase, 5 refusal
incl. plausible-but-out-of-corpus) pinned to the 48-chunk corpus via a
committed builder + .sig.json sidecar. New structural test guards re-stubbing.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

> If `git add data/eval/golden_curated.sig.json` reports the path is ignored, force-add it (`git add -f`) — the sidecar must be committed next to the golden.

---

## PR-E — MVP prompt tightening

### Task E1: Failing prompt assertions

**Files:**
- Test: `tests/test_eval_generation.py` (modify)

- [ ] **Step 1: Add the two new tests**

Append to `tests/test_eval_generation.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `py -3 -m pytest tests/test_eval_generation.py -q`
Expected: the two new tests FAIL (current prompt says «[1], [2] … там, где они уместны», has no `[N]`/«Каждое утверждение», and does not contain `IRRELEVANT_REFUSAL`). `test_system_prompt_matches_production` still passes.

### Task E2: Tighten both prompt constants (identically)

**Files:**
- Modify: `app/api/kb_mvp.py:406`
- Modify: `app/eval/generation_eval.py:20`

- [ ] **Step 1: Replace `_RAG_SYSTEM_PROMPT` in `app/api/kb_mvp.py`**

```python
_RAG_SYSTEM_PROMPT = (
    "Ты — помощник корпоративной базы знаний. Отвечай на русском. "
    "Используй ТОЛЬКО фрагменты из приведённого контекста и не добавляй фактов, которых в нём нет. "
    "Каждое утверждение в ответе сопровождай ссылкой на номер подтверждающего фрагмента в формате [N]. "
    "Если в контексте недостаточно данных, ответь ровно фразой: "
    "Не удалось найти в документах информацию для ответа."
)
```

- [ ] **Step 2: Replace `RAG_SYSTEM_PROMPT` in `app/eval/generation_eval.py` with the byte-identical string**

```python
# MUST stay byte-identical to app.api.kb_mvp._RAG_SYSTEM_PROMPT (drift-tested).
RAG_SYSTEM_PROMPT = (
    "Ты — помощник корпоративной базы знаний. Отвечай на русском. "
    "Используй ТОЛЬКО фрагменты из приведённого контекста и не добавляй фактов, которых в нём нет. "
    "Каждое утверждение в ответе сопровождай ссылкой на номер подтверждающего фрагмента в формате [N]. "
    "Если в контексте недостаточно данных, ответь ровно фразой: "
    "Не удалось найти в документах информацию для ответа."
)
```

- [ ] **Step 3: Run to verify all prompt tests pass**

Run: `py -3 -m pytest tests/test_eval_generation.py -q`
Expected: PASS (drift test + the two new tests + the existing refusal/judge tests).

### Task E3: Format and commit

- [ ] **Step 1: Format**

Run: `py -3 -m ruff check app/api/kb_mvp.py app/eval/generation_eval.py tests/test_eval_generation.py --fix; py -3 -m black app/api/kb_mvp.py app/eval/generation_eval.py tests/test_eval_generation.py`
Expected: no remaining lint errors.

- [ ] **Step 2: Regression-check the MVP surface**

Run: `py -3 -m pytest tests/test_eval_generation.py tests/test_kb_mvp.py -q`
Expected: PASS. (No test pins the old prompt wording — only the drift test references the constant.)

- [ ] **Step 3: Commit**

```bash
git add app/api/kb_mvp.py app/eval/generation_eval.py tests/test_eval_generation.py
git commit -m "feat(kb): per-claim citations + canonical refusal in MVP prompt

Tighten _RAG_SYSTEM_PROMPT: every claim must carry a [N] citation, and an
insufficient-context answer must be exactly IRRELEVANT_REFUSAL so the
deterministic refusal_correct metric is meaningful. Eval drift-pin updated in
lockstep; two new assertions added. MVP-only (v1 orchestrator is a queued
sibling); ships without a compare report by design (reversible, weightless).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## PR-R — Turnkey runbook (+ optional `sig` subcommand)

### Task R1: Runbook document

**Files:**
- Create: `docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md`

- [ ] **Step 1: Write the runbook**

```markdown
# Eval Baseline + Queued Gates — Runbook

> Run this the moment a real embedder + LLM exist. Until then, the curated
> golden + tightened prompt have already shipped; everything here is gated on
> a trustworthy baseline. Status is judged from git, not checkboxes.

## 0. Stand up a real embedder + LLM (pick one)

**Ollama (local, free):**
    winget install --id Ollama.Ollama -e
    ollama pull bge-m3 ; ollama pull qwen2.5:3b
    $env:KB_EMBEDDINGS_BACKEND="ollama"; $env:OLLAMA_EMBED_MODEL="bge-m3"
    $env:KB_LLM_PROVIDER="ollama"; $env:OLLAMA_MODEL="qwen2.5:3b"

**OpenAI-compatible API (needs BOTH an embeddings endpoint AND an LLM key):**
    $env:KB_EMBEDDINGS_BACKEND="api"; $env:EMBEDDINGS_API_BASE_URL="https://<host>/v1"
    $env:EMBEDDINGS_API_KEY="<key>"; $env:EMBEDDINGS_API_MODEL="<embed-model>"
    $env:DEEPSEEK_API_KEY="<key>"   # judge/generation (chat key ≠ embeddings)

## 1. Reindex under the real embedder, refresh the curated sidecar

    py -3 -m scripts.kb_cli reindex --embedder ollama --force-yes   # or openai-compatible
    py -3 -m scripts.build_curated_golden    # rewrites .sig.json (chunk-ids unchanged)

## 2. Baseline (the HARD gate)

    py -3 -m scripts.eval_rag generate --out var/data/eval/golden_auto.jsonl --limit 200
    py -3 -m scripts.eval_rag run --golden var/data/eval/golden_curated.jsonl `
      --out var/data/eval/baseline.json --judge
Keep `baseline.json` — every gate compares against it. The 48-chunk corpus is
small; consider ingesting more docs for a stabler baseline before trusting deltas.

## 3. Queued gates (keep iff the metric improves, else revert)

Each: apply the change, re-run `run ... --out var/data/eval/<task>.json`, then
`py -3 -m scripts.eval_rag compare var/data/eval/baseline.json var/data/eval/<task>.json`.

- **C — Russian reranker.** `app/services/kb_rerank.py` + `app/retriever/rerank.py`
  `DEFAULT_MODEL_NAME` → `BAAI/bge-reranker-v2-m3`; enable (`KB_RERANK_ENABLED=true`).
  Gate: mrr@k / hit@5 ↑ **and** latency acceptable. (~600 MB model download.)
- **D — top_k.** `py -3 -m scripts.eval_sweep --golden var/data/eval/golden_curated.jsonl --values 5,8,10,12 --judge`
  Pick argmax completeness without dropping faithfulness; set MVP `ask` top_k.
- **B — e5 on v1.** Reindex the v1 path under an e5 model with `VECTOR_E5_PREFIX=true`
  (no-op for MVP/bge-m3). Gate: recall@k / mrr@k ↑.

## Sidecar refresh note

There is no CLI to rewrite only the sidecar; `scripts/build_curated_golden.py`
re-emits it (chunk-ids are reindex-stable, only `embedder_name`/`dim` change).
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md
git commit -m "docs(eval): turnkey runbook for baseline + queued gates (C/D/B)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task R2 (OPTIONAL): `eval_rag sig` convenience subcommand

Only do this if a literal one-command sidecar refresh is wanted.

**Files:**
- Modify: `scripts/eval_rag.py`
- Test: `tests/test_eval_cli.py` (modify)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_cli.py`:

```python
def test_sig_subcommand_writes_sidecar(tmp_path, monkeypatch):
    import scripts.eval_rag as cli
    from app.eval.dataset import read_signature

    store, _ = _store_with_chunk(tmp_path)
    monkeypatch.setattr(cli, "get_store", lambda: store)
    golden = tmp_path / "g.jsonl"
    golden.write_text('{"instruction":"q","input":"","output":"a","meta":{"relevant_chunk_ids":[1]}}\n', encoding="utf-8")
    cli.main(["sig", "--golden", str(golden)])
    assert read_signature(golden) is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `py -3 -m pytest tests/test_eval_cli.py::test_sig_subcommand_writes_sidecar -q`
Expected: FAIL (`invalid choice: 'sig'`).

- [ ] **Step 3: Implement the subcommand**

In `scripts/eval_rag.py`, add the handler (after `cmd_compare`):

```python
def cmd_sig(args: argparse.Namespace) -> None:
    store = get_store()
    write_signature(Path(args.golden), compute_signature(store))
    print(f"Wrote signature sidecar for {args.golden}")
```

and register it in `build_parser` (after the `compare` parser):

```python
    sig = sub.add_parser("sig", help="(re)write a golden set's corpus-signature sidecar")
    sig.add_argument("--golden", default="data/eval/golden_curated.jsonl")
    sig.set_defaults(func=cmd_sig)
```

- [ ] **Step 4: Run to verify it passes**

Run: `py -3 -m pytest tests/test_eval_cli.py::test_sig_subcommand_writes_sidecar -q`
Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
py -3 -m ruff check scripts/eval_rag.py tests/test_eval_cli.py --fix; py -3 -m black scripts/eval_rag.py tests/test_eval_cli.py
git add scripts/eval_rag.py tests/test_eval_cli.py
git commit -m "feat(eval): eval_rag sig subcommand to refresh the signature sidecar

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

If R2 is done, update the runbook's "Sidecar refresh note" to mention `py -3 -m scripts.eval_rag sig`.

---

## Final verification

- [ ] **Run the full touched suite**

Run: `py -3 -m pytest tests/test_golden_curated.py tests/test_eval_generation.py tests/test_eval_dataset.py tests/test_eval_cli.py tests/test_kb_mvp.py -q`
Expected: PASS, offline.

- [ ] **Confirm branch state**

Run: `git log --oneline -4` and `git status --short`
Expected: PR-G, PR-E, PR-R commits present; working tree clean.

- [ ] **Integrate the branch** via superpowers:finishing-a-development-branch (push + PR, or merge per user preference).

---

## Out of scope / queued (do NOT do here)

- RU reranker default (C), top_k pick (D), e5-on-v1 (B) — measurement-gated; live behind the runbook.
- v1 `chat_orchestrator` prompt — separate symmetric sibling PR.
- Corpus expansion with new documents; two-path API merge; LoRA/DPO changes.
