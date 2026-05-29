# W4 — DPO Post-Training (closes G3)

**Date:** 2026-05-29
**Status:** Design — ready for implementation planning.
**Closes:** Gap **G3** from `docs/superpowers/specs/2026-05-25-ml-strengthening-pack-b-design.md` § Workstream 4 (DPO post-training / preference learning).
**Composes with:** W1 (`app/services/synthetic_qa.py`) for seed Q&A, W3 (`scripts/train_lora.py:format_prompt`) for prompt routing.
**Sequel to:** PR #559 (W3 RAG-aware fine-tuning).

---

## 1. Why this workstream

After W3 ships, the SFT adapter teaches the model **to ground answers in retrieved context with citations**. It does not teach the model **which of two grounded answers is preferred**. The pack-B++ spec estimates a +5–10 pp quality uplift from DPO over SFT-only baselines — comparable in magnitude to W3's RAG-grounding uplift, at ~one-fifth the dataset-generation cost (one teacher call per pair vs. two for synthetic Q&A).

Three concrete failure modes G3 prevents:

| Failure mode | DPO signal that fixes it |
|---|---|
| Model drops the `[doc_chunk:X]` citation suffix once it sounds confident | `rejected` = same answer with citation stripped |
| Model ignores retrieved context and pattern-matches from training data | `rejected` = generic ChatGPT-style closed-book answer |
| Model invents plausible-sounding fake citations | `rejected` = answer with fabricated `[doc_chunk:99]` |

---

## 2. Scope — one plan, three deliverables

Confirmed via brainstorming:

| ID | Deliverable | Touch points |
|---|---|---|
| W4-A | Synthetic preference dataset generator | `app/services/dpo_dataset.py` (pure logic), `scripts/generate_dpo_pairs.py` (CLI) |
| W4-B | DPO trainer (TRL-backed) | `scripts/train_dpo.py` |
| W4-C | Live feedback collection | `app/api/kb_feedback.py` (router), `app/services/kb_store.py` (+table, +methods), `app/api/kb_mvp.py` (router registration) |

All three ship in **one branch** under one PR (estimated ~600–700 LoC) following the W3 pattern of self-contained workstream PRs. If review pushes for smaller diffs, the plan's abort points permit a 2-PR split (datasets+endpoints first, then trainer + trl stub).

**NOT in scope (open separate issues if needed):**
- Measuring +5 pp DPO uplift on held-out RAGAS — owned by **W5**.
- Multi-rater preference aggregation, IRT-style modelling — over-engineering for MVP.
- Web-UI thumbs-up/down buttons in `data/www/index.html` — covered by W7 / W9 (Auto-Train UI) workstream when it lands.
- TIES / DARE adapter merging — explicit out-of-scope per `ROADMAP.md`.

---

## 3. Design decisions

### 3.1 Three reject strategies with Hamilton apportionment

`chosen` is always the seed answer with its `[doc_chunk:X]` citation intact (from W1). `rejected` is one of three strategies, distributed via Hamilton's largest-remainder method to keep the total exact across re-runs:

| Strategy | Share | Cost | Implementation |
|---|---|---|---|
| `NO_CITATION` | 40 % | 0 LLM calls | Regex strip `\s*\[doc_chunk:\d+\]\s*` from `chosen` — reuses `app.services.rag_dataset._strip_citations` (cross-workstream code reuse) |
| `GENERIC` | 30 % | 1 teacher call | Prompt asks teacher to answer the question **without** the retrieved chunk |
| `HALLUCINATION` | 30 % | 1 teacher call | Prompt asks teacher to answer **and invent a fake `[doc_chunk:9XX]` citation** |

Distribution implemented via `apportion_counts` already defined in `app/services/rag_dataset.py` — same Hamilton helper, second workstream now relies on it. (If the function moves to a shared utility module later, both W3 and W4 callers update together; no separate refactor needed for W4.)

### 3.2 Feedback shape — binary ±1 + optional `alternative_answer`

Per spec; rejected richer 1–5 grading because:
- Live thumbs-up/down has higher click-through than star widgets (well-documented in HCI literature).
- DPO formally takes a **pair**, not a ranked tuple. A binary signal maps natively.
- `alternative_answer` field captures the strongest signal (user-provided correct answer) at near-zero UI cost — one text input shown after thumbs-down.

### 3.3 Test strategy — stub trl + transformers (same pattern as `train_lora.py`)

Local dev has no `trl` / `transformers` / `peft` installed (per `CLAUDE.md` "Runtime conventions"). Tests use `tests/stubs/trl/` (new) and existing `tests/stubs/transformers/` shadows. Real ML libs are loaded only in CI (`-m integration` marker).

**Rationale:** `tests/stubs/` is the established repo convention (see `MEMORY.md` "Repo: test-stubs"). It keeps local TDD loops fast and deterministic, and tests double as **contract checks** for the stub vs. the real package signature.

---

## 4. Architecture

### 4.1 File layout

**Create:**
```
app/services/dpo_dataset.py            # ~250 LoC pure logic
scripts/generate_dpo_pairs.py          # ~150 LoC CLI
scripts/train_dpo.py                   # ~200 LoC CLI
app/api/kb_feedback.py                 # ~120 LoC FastAPI router
tests/stubs/trl/__init__.py            # ~80 LoC DPOConfig + DPOTrainer
tests/test_dpo_dataset.py
tests/test_dpo_dataset_strategies.py
tests/scripts/test_generate_dpo_pairs.py
tests/test_kb_feedback_store.py
tests/test_kb_feedback_api.py
tests/scripts/test_train_dpo.py
tests/test_train_dpo_integration.py    # @pytest.mark.integration
```

**Modify:**
```
app/services/kb_store.py               # +CREATE TABLE kb_feedback in _initialise_schema
                                       # +store_feedback() and iter_feedback_pairs() methods
app/api/kb_mvp.py                      # +app.include_router(kb_feedback.router)
requirements-runtime.txt               # +trl~=0.11 (optional dev-extras decision deferred to plan)
README.md / docs/legacy_README.md      # W4 usage example after W3 section
```

**Do NOT modify:**
```
app/services/synthetic_qa.py           # W1 stays stable; W4 composes via QAPair iterator
app/services/rag_dataset.py            # W3 stays stable; W4 reuses _strip_citations
scripts/train_lora.py                  # Only re-imports format_prompt
```

### 4.2 Composition diagram

```
W1 seeds.jsonl (QAPair{instruction, output_w_citation, source_chunk_id})
                            │
                            ▼
                scripts/generate_dpo_pairs.py
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
       NO_CITATION    GENERIC         HALLUCINATION
         (regex)    (teacher LLM)    (teacher LLM)
              │             │              │
              └─────────────┼──────────────┘
                            ▼
              DPOPair{prompt, chosen, rejected, meta}.to_jsonl_line()
                            │
                            ▼
                       dpo.jsonl
                            │
                            ▼
                  scripts/train_dpo.py
                  ├─ format_prompt (W3 reuse, --prompt-mode propagation)
                  ├─ chat_template via tokenizer
                  ├─ trl.DPOTrainer(model + SFT adapter, beta, lr)
                  └─ save DPO adapter

Orthogonal data path (Source 2 — live):
Chat UI → POST /api/kb/messages/{id}/feedback {rating, alternative_answer?}
                                  │
                                  ▼
                  kb_store.store_feedback() → kb_feedback table
                                  │
                                  ▼
                GET /api/kb/feedback/export
                  └─ kb_store.iter_feedback_pairs() (pairing logic)
                                  │
                                  ▼
                       Same DPOPair JSONL format
                                  │
                                  ▼
                       train_dpo.py --train live_pairs.jsonl
```

### 4.3 DPOPair output schema

```json
{
  "prompt": "<question>",
  "chosen": "<grounded answer with [doc_chunk:X]>",
  "rejected": "<failure-mode answer>",
  "meta": {
    "source": "synthetic|live",
    "strategy": "no_citation|generic|hallucination|live_alt|live_paired",
    "source_chunk_id": 42,
    "feedback_ids": ["<uuid>", ...]
  }
}
```

Top-level `prompt/chosen/rejected` keys match `trl.DPOTrainer` dataset contract directly — no transform pass needed before training.

---

## 5. Schema additions to `kb_store`

Added inside `_initialise_schema`, after the existing `CREATE TABLE IF NOT EXISTS kb_messages` statement:

```sql
CREATE TABLE IF NOT EXISTS kb_feedback (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL
        REFERENCES kb_conversations(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL
        REFERENCES kb_messages(id) ON DELETE CASCADE,
    user_id TEXT,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    comment TEXT,
    alternative_answer TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kb_feedback_message
    ON kb_feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_kb_feedback_rating_created
    ON kb_feedback(rating, created_at);
```

**Idempotent migration:** the MVP path uses `CREATE TABLE IF NOT EXISTS` in `_initialise_schema` — running servers will pick up the new table on next start with no operator action. No Alembic step (kb_mvp does not use Alembic; the `/api/v1` multi-tenant path does, and W4 does not extend it).

**No `UNIQUE(message_id, user_id)`:** preferences may evolve. Pairing logic uses the most-recent rating per `(message_id, user_id)` rather than a single canonical row.

---

## 6. API contracts

### 6.1 `POST /api/kb/messages/{message_id}/feedback`

```
Auth:    Bearer KB_API_KEY (same single-tenant pattern as the rest of /api/kb)
Body:    {"rating": 1, "comment": "...", "alternative_answer": "..."}
         rating  required, must be -1 or 1
         comment optional, max 2000 chars
         alternative_answer optional, max 4000 chars

Response 201:
         {"id": "<uuid>", "created_at": "2026-05-29T..."}

Errors:
  400 — rating not in {-1, 1} OR field length exceeded
  401 — missing/wrong KB_API_KEY
  404 — message_id not in kb_messages
```

### 6.2 `GET /api/kb/feedback/export`

```
Auth:    Bearer KB_API_KEY
Query:   since=ISO-8601 (optional)
         min_pairs=int (optional; default 0 — emit whatever is available)
Headers: Accept: application/x-ndjson

Response 200 (application/x-ndjson):
         <one DPOPair JSONL per line>
         Header `X-DPO-Pairs-Count` with total count emitted
         Header `Warning` if `min_pairs > available` (does not fail)
```

### 6.3 Pairing logic for live export

Implemented in `kb_store.iter_feedback_pairs()`. The `prompt` field for each emitted pair is the **immediately-preceding `role='user'` message** in the same `conversation_id` (resolved via a single SQL window function or `ORDER BY created_at DESC LIMIT 1` subquery). If no preceding user message exists (orphaned assistant message — should not happen but guard anyway), the pair is skipped with a debug log.

Pseudocode:

```python
for (message_id, user_id), rows in group_feedback_by_message_user():
    rated = sorted(rows, key=lambda r: r.created_at, reverse=True)
    latest = rated[0]
    assistant_msg = kb_messages[message_id]              # role='assistant'
    user_prompt = preceding_user_message(assistant_msg)  # role='user', same conversation_id
    if user_prompt is None:
        continue  # orphaned assistant message — defensive skip
    assistant_text = assistant_msg.content

    if latest.rating == 1:
        # User approved this assistant answer
        chosen = assistant_text
        if latest.alternative_answer:
            # Stronger: user *also* provided their version — emit two pairs
            yield DPOPair(user_prompt, chosen=latest.alternative_answer, rejected=assistant_text,
                          meta={"source": "live", "strategy": "live_alt"})
            continue
        # Else: look for a recent -1 with alt to use as rejected
        downvote = next((r for r in rated if r.rating == -1 and r.alternative_answer), None)
        if downvote:
            yield DPOPair(user_prompt, chosen=chosen, rejected=downvote.alternative_answer,
                          meta={"source": "live", "strategy": "live_paired"})
    elif latest.rating == -1 and latest.alternative_answer:
        yield DPOPair(user_prompt, chosen=latest.alternative_answer, rejected=assistant_text,
                      meta={"source": "live", "strategy": "live_alt"})
    # else: insufficient signal → skip silently
```

Important: pairs are emitted **at most once per `(message_id, user_id)`** to avoid biasing DPO toward over-rated messages.

---

## 7. Error handling

| Scenario | `generate_dpo_pairs.py` | `train_dpo.py` | endpoints |
|---|---|---|---|
| Teacher LLM 5xx / timeout | Retry × 3 w/ exponential backoff → skip + log warning | n/a | n/a |
| Malformed W1 JSONL line | Warning + skip (mirrors W3 `_load_seeds`) | n/a | n/a |
| Estimated cost > budget | `SystemExit` unless `--yes` | n/a | n/a |
| `trl` not installed | n/a (no trl needed for dataset gen) | `SystemExit("install: pip install trl~=0.11")` | n/a |
| Empty input dataset | log info + exit 0 | log warning + exit 0 | n/a |
| message_id not found | n/a | n/a | 404 |
| Empty export result | n/a | n/a | 200 + `[]` + log info |
| SQLite locked | n/a | n/a | Retry × 3 → 503 |

### Cost guard (mirrors W1 / W3 budget enforcement)

```python
def estimate_cost(seed_count: int, strategy_mix: dict[RejectStrategy, int]) -> float:
    teacher_calls = sum(c for s, c in strategy_mix.items() if s != RejectStrategy.NO_CITATION)
    return teacher_calls * 0.0005  # DeepSeek-V3 baseline ~$0.50 / 1000

# in main():
if estimate_cost(...) > args.max_cost_usd and not args.yes:
    raise SystemExit(f"Estimated ${cost:.2f} > ${args.max_cost_usd}. Pass --yes to override.")
```

---

## 8. Testing strategy

### 8.1 Pyramid

| Tier | Count | Runtime | Markers |
|---|---|---|---|
| Unit — pure logic | ~20 | < 0.5 s | (none — runs always) |
| Unit — endpoints (FastAPI TestClient) | ~8 | ~1 s | (none) |
| End-to-end CLI smoke (tmp_path SQLite) | ~5 | ~2 s | (none — stub-backed) |
| Integration — real TRL on tiny model | ~3 | ~3 min | `@pytest.mark.integration` |

### 8.2 Stub `trl` minimal surface

Mirrors real `trl.DPOTrainer` 0.11+ signature:

```python
# tests/stubs/trl/__init__.py
from dataclasses import dataclass
from typing import Any

@dataclass
class DPOConfig:
    output_dir: str
    beta: float = 0.1
    learning_rate: float = 5e-7
    per_device_train_batch_size: int = 4
    num_train_epochs: int = 1
    max_length: int = 1024
    max_prompt_length: int = 512
    logging_steps: int = 10
    save_steps: int = 100
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False

class DPOTrainer:
    train_calls: list[dict[str, Any]] = []

    def __init__(self, model=None, ref_model=None, args=None,
                 train_dataset=None, eval_dataset=None,
                 tokenizer=None, peft_config=None, **kwargs):
        self.model, self.args = model, args
        self.train_dataset, self.peft_config = train_dataset, peft_config

    def train(self) -> None:
        DPOTrainer.train_calls.append({
            "model": self.model,
            "beta": self.args.beta if self.args else None,
            "dataset_size": len(self.train_dataset) if self.train_dataset else 0,
        })

    def save_model(self, output_dir: str) -> None:
        import pathlib
        p = pathlib.Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "adapter_config.json").write_text("{}")
```

### 8.3 Stub-shadow pattern

`tests/conftest.py` already prepends `tests/stubs/` to `sys.path`. If real `trl` is installed in CI, it wins. Locally, the stub answers. **Tests do not need `@pytest.mark.skip`**.

For `train_dpo.py` itself, tests call `from scripts.train_dpo import parse_args, main` etc. The import resolves `trl` to whichever copy is on path. The `pytest.importorskip("trl")` pattern is reserved for the `@pytest.mark.integration` test that needs the **real** library.

### 8.4 Coverage targets

| Module | Target | Rationale |
|---|---|---|
| `dpo_dataset.py` | 95 %+ | Pure logic, easy with fakes |
| `kb_feedback.py` (router) | 90 %+ | FastAPI TestClient + tmp sqlite |
| `kb_store.store_feedback/iter_feedback_pairs` | 90 %+ | SQL pairing logic — critical to DPO quality |
| `generate_dpo_pairs.py` | 80 %+ | CLI + I/O, argparse-driven |
| `train_dpo.py` | 60 %+ | TRL-bound; balance covered by integration test |

---

## 9. Acceptance criteria

- [ ] `py -3 -m pytest tests/test_dpo_dataset.py tests/test_dpo_dataset_strategies.py tests/test_kb_feedback_store.py tests/test_kb_feedback_api.py tests/scripts/test_generate_dpo_pairs.py tests/scripts/test_train_dpo.py -v` — all pass.
- [ ] `py -3 -m pytest -q --ignore=backend` — no regressions vs. `main`.
- [ ] `py -3 -m ruff check . && py -3 -m black --check .` — clean.
- [ ] `py -3 -m scripts.generate_dpo_pairs --seeds … --output dpo.jsonl --target-pairs 40 --yes` writes a valid JSONL where:
  - Every line has top-level `prompt / chosen / rejected`.
  - Strategy distribution matches 40 / 30 / 30 within ±1 sample.
  - Every line has `meta.strategy ∈ {no_citation, generic, hallucination}`.
- [ ] `POST /api/kb/messages/{id}/feedback` with valid body returns 201 and persists; invalid rating returns 400.
- [ ] `GET /api/kb/feedback/export` returns NDJSON whose lines can be passed directly to `train_dpo.py --train -`.
- [ ] `scripts/train_dpo.py --base-model stub --train dpo.jsonl --sft-adapter <path> --output adapters/my-dpo --prompt-mode rag --max-steps 1` runs to completion under the stub.
- [ ] PR description points to spec § Workstream 4 and notes the G3 metric improved (preference accuracy; held-out faithfulness — measured by W5).

---

## 10. Out of scope (parking lot)

- **W5 RAGAS evaluation** — measures the resulting DPO uplift. Owns the +5 pp number.
- **Web-UI thumbs buttons** in `data/www/` — separate visual change, lands with W7 or W9.
- **Multi-rater consensus** — current pairing assumes one rating per `(message, user)`.
- **TIES / DARE adapter merging** of SFT and DPO adapters — explicit ROADMAP anti-feature.
- **Embedding fine-tuning** to improve retrieval quality before DPO — W6.

---

## 11. Estimated effort

- Sprint 1 (`dpo_dataset.py` module + 3 reject strategies + builder): ~3 h.
- Sprint 2 (`generate_dpo_pairs.py` CLI + cost guard + resume): ~1.5 h.
- Sprint 3 (`kb_feedback` schema + store methods + endpoints): ~2 h.
- Sprint 4 (`tests/stubs/trl` + `scripts/train_dpo.py` + integration test scaffolding): ~3 h.
- Sprint 5 (docs + PR): ~30 min.
- **Total: 10–12 h of focused TDD work, comparable to W3.**

---

## 12. Open questions

None blocking. Two questions deferred to plan-time choice (not design):

1. **Should `trl` move to a `requirements-ml-extras.txt`?** The W3 PR already adds new optional ML deps via the main `requirements-runtime.txt`. Keep consistent unless reviewer pushes back.
2. **`POST /feedback` returning the persisted row vs. just `{id}`** — both are fine. Plan to choose; current draft chooses `{id, created_at}` for minimal API surface.
