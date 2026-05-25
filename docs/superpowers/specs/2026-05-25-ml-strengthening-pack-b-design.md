# ML Strengthening — Pack B Design

**Date:** 2026-05-25
**Author scope:** technical design for ML capability uplift in KB.AI
**Status:** Design document. Subordinate to `2026-05-22-project-vision-design.md` (LoRA as main moat).
**Decision context:** Technical analysis of existing ML stack identified 7 critical gaps and 11 secondary improvements. Pack B (8-10 weeks) bundles the 9 most valuable items into a coherent ML platform uplift.

> Этот документ описывает **технические доработки ML-стека**, не привязываясь к бизнес-срокам и финансам. Реализация может быть растянута/сжата под доступные ресурсы. Документ — это **что и как делать**, не **когда успеть**.

---

## 1. Context

KB.AI уже имеет нетривиальный LoRA-пайплайн:

| Компонент | Файл | LoC | Состояние |
|---|---|---|---|
| PEFT LoRA training | `scripts/train_lora.py` | 389 | QLoRA 4-bit, FP16/BF16, gradient checkpointing |
| Dataset validation | `scripts/validate_dataset.py` | 340 | Token stats, duplicates, overflow checks |
| PEFT→GGUF conversion | `scripts/convert_lora_to_gguf.py` | 89 | Subprocess wrapper for `llama_cpp.convert_lora` |
| Evaluation (EM, ROUGE-L) | `scripts/eval_lora.py` | 227 | Threshold-checks for CI |
| Adapter FS registry | `app/llm/lora_runtime.py` | 252 | manifest.json + payload, list/load/unload |
| Async runtime manager | `app/services/lora_manager.py` | 162 | Scaling validation [0, 10], exceptions |
| llama.cpp provider | `app/llm/llama_cpp_provider.py` | 289 | GGUF + PEFT load hooks |
| REST API | `/api/v1/lora/*`, `/llm/adapters/*` | — | Hot-load via name |

**This is already stronger than most opensource RAG projects.** AnythingLLM, Onyx, LangChain templates do not ship LoRA. GigaChat and YandexGPT expose closed APIs. The right question is therefore **not "should we strengthen LoRA"** but **"what delta in the LoRA stack makes a customer say 'wow'"**. This document specifies that delta.

## 2. Critical gaps in current ML stack

| # | Gap | Impact |
|---|---|---|
| G1 | No automatic training-dataset generation from corpus | Customer must hand-write JSONL — blocking adoption |
| G2 | No RAG-aware fine-tuning prompts | -15-25% faithfulness on domain tasks |
| G3 | No DPO / preference learning | Missing +5-10% quality available "for free" via reformulation |
| G4 | Evaluation uses EM + ROUGE-L (weak for RAG) | No defensible quality numbers for sales |
| G5 | No embedding fine-tuning (only LLM is tuned) | Retrieval stays generic, weakens entire pipeline |
| G6 | No continual / incremental fine-tuning | Each retrain starts from scratch — wasted effort |
| G7 | No Auto-Train UI — only CLI | LoRA stays a dev-only feature, not productized |

## 3. Pack B scope — 9 workstreams

Pack B implements all 7 critical gaps plus 2 supporting items (MLflow tracking, Golden-set regression). Pack A (gaps G1, G4, G7 + regression test) is **strict subset** of Pack B.

### Workstream 1: Synthetic data generation (closes G1)

**New file:** `scripts/generate_synthetic_qa.py`

**Behaviour:**
- Input: corpus path (chunks already in Qdrant or KB SQLite) + teacher model config + output JSONL path
- For each chunk, query teacher LLM (default: DeepSeek-V3 via existing `app/services/kb_llm.py` provider) to produce N diverse Q&A pairs
- Filter outputs:
  - Length constraints (instruction 10-200 chars; output 30-2000 chars)
  - Self-consistency check (re-generate and compare)
  - Toxicity / refusal filter
- Output: JSONL compatible with `train_lora.py` and `validate_dataset.py`

**Prompt design** (one of several templates, randomised per chunk):
```
Ты — эксперт по составлению обучающих примеров. На основе следующего фрагмента документа сгенерируй ОДИН вопрос, который мог бы задать сотрудник компании, и точный ответ с указанием источника.

Фрагмент [doc:{doc_id}, page:{page}]:
{chunk_text}

Формат JSON:
{"instruction": "...", "input": "", "output": "Ответ: ... Источник: [doc:{doc_id}, page:{page}]"}
```

**Generation modes:**
- `single` — один Q&A на чанк
- `paraphrase` — 3 версии одного вопроса для аугментации
- `multi-hop` — вопрос, требующий комбинации 2-3 чанков

**Cost guard:** estimated tokens before running; refuse if > configurable budget. DeepSeek-V3 baseline: ~$0.50 per 1000 generated Q&A pairs.

**Acceptance:**
- 1000 Q&A pairs generated from 500-chunk corpus in <30 min
- 95%+ pairs pass `validate_dataset.py` without warnings
- Output matches existing prompt template format

---

### Workstream 2: Architecture-aware training (extends `train_lora.py`)

**Current limitation:** hardcoded `PROMPT_TEMPLATE = "<s>[INST] {instruction}\n{input} [/INST]\n"` (Mistral/Llama-2 format).

**Changes to `scripts/train_lora.py`:**

1. **Auto chat template detection** via `tokenizer.chat_template` (HuggingFace standard):
   ```python
   def _format_prompt(tokenizer, instruction, context, output=None):
       if tokenizer.chat_template:
           messages = [{"role": "user", "content": f"{instruction}\n{context}" if context else instruction}]
           if output is not None:
               messages.append({"role": "assistant", "content": output})
               return tokenizer.apply_chat_template(messages, tokenize=False)
           return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
       # Fallback to legacy Mistral format
       return PROMPT_TEMPLATE_LEGACY.format(instruction=instruction, input=context or "")
   ```

2. **Target modules lookup table** (new `app/llm/target_modules.py`):
   ```python
   TARGET_MODULES = {
       "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
       "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
       "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
       "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
       "gpt2": ["c_attn", "c_proj"],  # TinyLlama may inherit this
   }
   def resolve_target_modules(model_config):
       arch = model_config.model_type.lower()
       return TARGET_MODULES.get(arch, "all-linear")
   ```
   When user passes `--target-modules` it overrides lookup.

3. **FSDP support** for 70B+ models via `accelerate launch`:
   - New `scripts/train_lora_fsdp.py` (or `--use-fsdp` flag) that uses `accelerate.FullyShardedDataParallelPlugin`
   - Config file `accelerate_fsdp.yaml` for distributed training across multiple GPUs
   - Tested on Llama-3-70B with 2x A100-80GB

**Supported base models in Pack B:**
- TinyLlama-1.1B / Phi-3-mini (3.8B) — CPU + small GPU
- Saiga-Llama-3-8B / Qwen2.5-7B — single GPU (24GB VRAM with QLoRA)
- Qwen2.5-14B — single GPU (36GB)
- Llama-3-70B / Mistral-Large — multi-GPU FSDP

**Acceptance:**
- One unified `train_lora.py` works for all 4 architectures without changes to user-supplied args
- Training Llama-3-70B with FSDP on 2x A100-80GB completes 1 epoch on 1000 examples in <2h

---

### Workstream 3: RAG-aware fine-tuning (closes G2)

**New prompt template** (per-mode, configurable via env):
```python
PROMPT_TEMPLATE_RAG = """<system>
Ответь на вопрос, используя контекст и свои знания. Если контекст релевантен — приоритизируй его. Указывай источник цитаты в формате [doc:X, page:Y].
</system>

Контекст:
{retrieved_context}

Вопрос: {instruction}"""
```

**New file:** `scripts/generate_rag_dataset.py` (composes with G1's synthetic generator):

For each generated Q&A:
1. Retrieve top-3 chunks via existing `KnowledgeBaseStore.search()` using the question
2. Build 4 training variants:
   - **Relevant context + correct answer** (positive sample, ~70%)
   - **Irrelevant context + answer "Не удалось найти в документах"** (negative sample, ~15%)
   - **Partially relevant context + answer with cautious citation** (~10%)
   - **Empty context + general knowledge answer** (~5%)

This teaches the model to:
- Use context when provided
- Refuse when context is insufficient
- Distinguish relevant vs. irrelevant chunks
- Generate citations in the required format

**Changes to `train_lora.py`:** accept `--prompt-mode {generic, rag}` parameter; when `rag`, expects dataset to have `retrieved_context` field.

**Acceptance:**
- Faithfulness score (RAGAS) improves by ≥10pp vs. baseline SFT on held-out test
- Refusal rate on out-of-corpus questions ≥80%

---

### Workstream 4: DPO post-training (closes G3)

**New file:** `scripts/train_dpo.py`

**Library:** `trl` (HuggingFace TRL — official DPO implementation)

**Inputs:**
- Base model + SFT-trained LoRA adapter (must exist before DPO)
- Preference dataset: JSONL with `prompt`, `chosen`, `rejected` fields
- Hyperparameters: `beta` (DPO temperature, default 0.1), `lr` (default 5e-7, much lower than SFT)

**Outputs:**
- DPO-tuned adapter (replaces SFT adapter or merges via TIES — configurable)
- Metrics: implicit reward gap, preference accuracy on validation

**Two sources of preference pairs:**

**Source 1: Synthetic via teacher.** `scripts/generate_dpo_pairs.py`:
- For each question in synthetic Q&A dataset:
  - `chosen` = answer with explicit citation [doc:X, page:Y]
  - `rejected` = either (a) same answer without citation, (b) generic ChatGPT-style answer ignoring context, (c) hallucination (teacher invents fake citation)
- 1000 pairs per ~$1 in DeepSeek API costs

**Source 2: Live feedback collection.** New table `kb_feedback`:
```sql
CREATE TABLE kb_feedback (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES kb_conversations(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL REFERENCES kb_messages(id),
    user_id TEXT,
    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    comment TEXT,
    alternative_answer TEXT,
    created_at TEXT NOT NULL
);
```

New endpoints:
- `POST /api/kb/messages/{id}/feedback` — rate ±1 with optional comment + alternative
- `GET /api/kb/feedback/export` — export as DPO-compatible JSONL (chosen = higher rated, rejected = lower rated)

**Acceptance:**
- 1000 synthetic preference pairs generated in <20 min for ~$1
- DPO improves answer quality (LLM-as-judge eval) by ≥5pp over SFT-only baseline
- `/feedback` endpoint persists and exports correctly

---

### Workstream 5: RAGAS evaluation (closes G4)

**New file:** `scripts/eval_ragas.py`

**Library:** `ragas` (PyPI)

**Metrics computed (via LLM-as-judge, default DeepSeek):**
- **Faithfulness**: ответ соответствует контексту (no hallucinations)
- **Answer relevance**: ответ адекватен вопросу
- **Context relevance**: извлечённый контекст релевантен вопросу
- **Context recall**: всё нужное было извлечено
- **Context precision**: топ-к ранжирован корректно

**Input dataset:** JSONL with `question`, `ground_truth`, `answer`, `contexts` fields. Generated by running the existing `/api/kb/ask` endpoint on the eval set.

**Output:** JSON + Markdown report (same structure as `eval_lora.py`); thresholds via `--min-faithfulness`, `--min-answer-relevance`, etc.

**Replaces** EM/ROUGE-L in CI gates (old metrics deprecated but kept as backward compatibility).

**Acceptance:**
- Full RAGAS eval on 100-question test set completes in <10 min
- Baseline numbers established for: generic SFT, RAG-aware SFT, DPO-tuned — used as sales material

---

### Workstream 6: Embedding fine-tuning (closes G5)

**New file:** `scripts/train_embedder.py`

**Library:** `sentence-transformers`

**Base models supported:**
- `intfloat/multilingual-e5-small` (current default, 384-dim)
- `intfloat/multilingual-e5-base` (768-dim)
- `BAAI/bge-m3` (1024-dim, multi-functional: dense + sparse + multi-vector)

**Training data:** triplet pairs `(anchor, positive, negative)`:
- **anchor** = question (from synthetic Q&A or feedback)
- **positive** = chunk that contains the answer
- **negative** = randomly sampled chunk from same corpus (hard negatives via BM25 mismatch)

Generation: `scripts/generate_triplets.py` runs on synthetic Q&A from G1, using the `[doc:X, page:Y]` annotations to identify ground-truth positive chunks.

**Loss:** Multiple Negatives Ranking Loss (`MultipleNegativesRankingLoss`) — standard for semantic search.

**Optional:** **Matryoshka Representation Learning** for adaptive dimensionality (`MatryoshkaLoss` over [64, 128, 256, 384] dims) — allows downstream choice of speed/quality tradeoff.

**Integration with KB stack:**
- Trained embedder saved to `var/embedders/<name>/` with manifest
- `app/services/kb_embeddings.py` extended with `local` backend pointing to local model
- `KB_EMBEDDINGS_BACKEND=local`, `KB_EMBEDDINGS_LOCAL_PATH=var/embedders/my-domain` activates

**Important constraint:** changing embedder invalidates existing chunk embeddings. Reindex required. The new `reindex_service.py` should accept `--embedder-changed` flag for full reembedding.

**Acceptance:**
- Trained embedder improves NDCG@10 by ≥0.1 over baseline `multilingual-e5-small` on domain corpus
- Reindex pipeline successfully migrates 10k chunks in <1h

---

### Workstream 7: Continual / incremental fine-tuning (closes G6)

**Changes to `scripts/train_lora.py`:**

New flags:
- `--resume-from-adapter PATH` — initialise from existing PEFT adapter weights
- `--continual-lr-factor 0.3` — multiplier applied to base learning rate (lower for stability)
- `--ewc-lambda 0.0` — optional EWC (Elastic Weight Consolidation) regularisation to prevent catastrophic forgetting

**EWC implementation:**
- Compute Fisher Information Matrix on the **old** dataset before continual training
- Add to loss: `L_total = L_new + ewc_lambda * sum(F_i * (theta_i - theta_old_i)^2)`
- Helper: `scripts/compute_fisher.py` — runs once after each successful training run, saves `fisher.pt` alongside the adapter

**Workflow:**
1. Train adapter v1 from scratch — saves adapter + Fisher matrix
2. New domain data arrives
3. Run `train_lora.py --resume-from-adapter v1 --train new_data.jsonl --ewc-lambda 0.1`
4. Produces adapter v2 that retains v1 knowledge AND learns new data

**Acceptance:**
- Continual training v1→v2 retains ≥95% of v1's accuracy on v1's held-out test
- Continual training learns new patterns from v2 data (accuracy on v2 test set ≥80% of fresh-trained baseline)

---

### Workstream 8: MLflow experiment tracking

**Integration points:**
- `scripts/train_lora.py`: log hyperparameters at start; log metrics from `JsonLogCallback` to MLflow; log adapter artefact at end
- `scripts/eval_lora.py` and `scripts/eval_ragas.py`: log eval metrics tied to the training run
- `scripts/train_dpo.py`, `scripts/train_embedder.py`: same pattern

**Config:**
- `MLFLOW_TRACKING_URI` env var (default: `./var/mlflow` for local file backend)
- For team setups: point to remote MLflow server (free PyPI package, runs in 1 docker container)

**UI surface:**
- Operations Console gets new tab "Experiments" — iframe to MLflow UI on `:5000`
- Compare runs, see metric trajectories, download adapter artefacts

**Acceptance:**
- All 4 training scripts log to MLflow without code changes for user
- Comparing 3 runs of `train_lora.py` shows divergent loss curves and final EM/ROUGE/RAGAS metrics

---

### Workstream 9: Auto-Train UI (closes G7)

**Frontend:** new tab in Operations Console (`data/www/admin.html` OR `frontend/src/pages/`).

**5-step wizard:**

1. **Choose corpus** — dropdown of available KB instances, or upload new ZIP of documents
2. **Generate dataset** — button "Generate Q&A from corpus" → calls Workstream 1 → progress bar with token count + cost estimate
3. **Choose base model** — preset cards:
   - "Быстрый" (TinyLlama, 30 min train, baseline quality)
   - "Сбалансированный" (Saiga-8B QLoRA, 2-4h train, good quality) — DEFAULT
   - "Качественный" (Qwen2.5-14B QLoRA, 6-12h train, top single-GPU quality)
   - "Максимум" (Llama-3-70B FSDP, requires multi-GPU, 24-48h train)
4. **Training pipeline** — checkboxes:
   - ☑ SFT (always on)
   - ☑ RAG-aware (default on)
   - ☑ DPO (requires synthetic preference data — generated automatically)
   - ☐ Embedder fine-tuning (separate, slower job)
5. **Run + monitor** — WebSocket stream of progress; live RAGAS scores after each epoch; "abort" button

**Backend:** new `app/services/training_orchestrator.py`:
- Job queue (SQLite-backed initially, Celery-ready)
- Endpoints: `POST /api/v1/training/jobs`, `GET /api/v1/training/jobs/{id}`, `WS /api/v1/training/jobs/{id}/stream`, `DELETE` (cancel)
- Hooks into existing `scripts/*` as subprocesses with structured stdout

**Acceptance:**
- Non-ML user can train Saiga-8B QLoRA adapter on their own corpus end-to-end without touching CLI
- Training job survives backend restart (resumable from last checkpoint)
- RAGAS scores visible before/after deployment

---

### Workstream 10 (supporting): Golden regression test

**New file:** `scripts/regression_test.py`

**Behaviour:**
- Maintains a curated `data/golden_qa/<workspace>/golden.jsonl` — 50-100 critical Q&A pairs that MUST work after any adapter change
- Triggered automatically on adapter hot-load via `/api/v1/llm/adapters/hot-load`
- Runs the questions through the live `/api/kb/ask` endpoint with the new adapter
- Compares answers to golden using RAGAS faithfulness + semantic similarity (cosine on embedder)
- If regression > threshold (e.g. -5pp faithfulness) — refuses to activate adapter, returns HTTP 409 with diff report

**Integration:** `LoraRuntimeManager.load_adapter()` calls regression check before `set_active_adapter()`.

**Acceptance:**
- Hot-load is rejected when synthetic regression is introduced (test: load adapter trained on irrelevant data, expect rejection)
- Hot-load passes when adapter quality is preserved

## 4. Data flow

```
[Customer documents]
      │
      ├──> [Existing ingest pipeline → Qdrant + KB SQLite]
      │
      ├──> [W1: generate_synthetic_qa.py] ──> synthetic_qa.jsonl
      │       │
      │       ├──> [W3: generate_rag_dataset.py] ──> rag_qa.jsonl
      │       │       │
      │       │       └──> [train_lora.py --prompt-mode rag] ──> SFT adapter (v1)
      │       │
      │       ├──> [generate_dpo_pairs.py via teacher] ──> dpo_pairs.jsonl
      │       │       │
      │       │       └──> [W4: train_dpo.py --resume-from-adapter v1] ──> DPO adapter (v2)
      │       │
      │       └──> [generate_triplets.py] ──> triplets.jsonl
      │               │
      │               └──> [W6: train_embedder.py] ──> domain embedder
      │
      ├──> [W7: train_lora.py --resume-from-adapter v2 + new_data] ──> v3 (continual)
      │
      ├──> [W5: eval_ragas.py] ──> RAGAS scores per adapter
      │
      ├──> [W10: regression_test.py] ──> gate adapter activation
      │
      ├──> [W8: MLflow] ──> all runs tracked
      │
      └──> [W9: Auto-Train UI] ──> orchestrates W1-W7 visually
```

## 5. Dependencies and order

Workstreams have explicit ordering due to data dependencies:

1. **W1 (synthetic data)** — first; everything else depends on it
2. **W2 (architecture-aware)** — first; refactor existing code, enables all training
3. **W5 (RAGAS)** — second; needed to measure W3, W4, W6, W7 improvements
4. **W6 (embedder)** — third; runs in parallel with W3
5. **W3 (RAG-aware)** — fourth; depends on W1 + W2
6. **W4 (DPO)** — fifth; depends on W3 (DPO needs SFT base)
7. **W7 (continual)** — sixth; depends on W4
8. **W8 (MLflow)** — can be added anytime in parallel
9. **W9 (Auto-Train UI)** — last; wraps everything; depends on W1-W7 being functional
10. **W10 (regression)** — added in parallel with W5

## 6. Files to be created or modified

**New files:**
- `scripts/generate_synthetic_qa.py`
- `scripts/generate_rag_dataset.py`
- `scripts/generate_dpo_pairs.py`
- `scripts/train_dpo.py`
- `scripts/train_embedder.py`
- `scripts/generate_triplets.py`
- `scripts/eval_ragas.py`
- `scripts/compute_fisher.py`
- `scripts/regression_test.py`
- `scripts/train_lora_fsdp.py` (or FSDP support in `train_lora.py`)
- `app/llm/target_modules.py`
- `app/services/training_orchestrator.py`
- `app/api/v1/training.py` (new routes)
- `alembic/versions/XXXX_kb_feedback.py` (migration for feedback table)
- Frontend: training wizard page (5 steps)
- `accelerate_fsdp.yaml`

**Modified files:**
- `scripts/train_lora.py` — auto chat template, architecture-aware target modules, `--resume-from-adapter`, `--ewc-lambda`, `--prompt-mode {generic, rag}`, MLflow integration
- `scripts/eval_lora.py` — MLflow integration; mark EM/ROUGE as deprecated
- `app/services/kb_embeddings.py` — `local` backend
- `app/services/lora_manager.py` — regression check before activation
- `app/services/reindex_service.py` — `--embedder-changed` flag
- `app/api/kb_mvp.py` — `/messages/{id}/feedback`, `/feedback/export`
- `requirements-train.txt` — add `trl`, `ragas`, `mlflow`, `sentence-transformers[train]`, `accelerate`
- `requirements-runtime.txt` — add `mlflow-skinny` (for runtime logging only)

## 7. Out of scope (Pack C territory)

Explicitly excluded — to be addressed in a separate future spec:

- Multi-adapter composition (loading 2+ LoRAs with different scales simultaneously)
- Structured outputs via `outlines` / llama.cpp grammar
- Quantization-Aware Training (QAT)
- Knowledge distillation from large teacher to small student
- TIES / DARE adapter merging
- Distributed training beyond FSDP (DeepSpeed ZeRO-3, multi-node)
- Model registry with hot-swap base model

## 8. Risks and mitigation

| Risk | Mitigation |
|---|---|
| Synthetic data quality drift (teacher hallucinations) | Self-consistency filter in W1; spot-check via human review of 50 samples |
| RAGAS LLM-as-judge bias | Run with 3 different judges (DeepSeek, Groq, GPT-4); report avg + variance |
| DPO collapses on small preference datasets | Use β=0.1 (conservative); require ≥500 preference pairs; monitor implicit reward gap |
| Catastrophic forgetting in continual learning | EWC regularisation; mandatory regression test (W10) gates activation |
| FSDP setup complexity for 70B | Document multi-GPU setup in dedicated runbook; provide reference `accelerate_fsdp.yaml` |
| Auto-Train UI exposes hyperparameters that confuse non-ML users | Presets only ("Быстрый/Сбалансированный/Качественный/Максимум"); advanced mode hidden behind toggle |
| MLflow remote server overhead | Default to local file backend `./var/mlflow`; remote optional |

## 9. Success metrics (post Pack B)

Quantitative — measured via RAGAS on a held-out test set per workspace. The numbers below are **expected** ranges based on published literature; real baselines must be measured on the actual customer corpus at the start of Pack B implementation. Treat the column "Expected target" as a delta over measured baseline, not absolute values.

| Metric | Expected baseline (generic SFT) | Expected target after Pack B | Improvement delta |
|---|---|---|---|
| Faithfulness | 0.60–0.70 | 0.80+ | +15–20pp |
| Answer relevance | 0.65–0.75 | 0.85+ | +10–15pp |
| Context recall | 0.50–0.60 | 0.75+ | +15–20pp |
| Refusal rate on out-of-corpus questions | 20–40% | 80%+ | +40–60pp |

**Measurement protocol:**
1. Before any Pack B workstream starts, run `eval_ragas.py` against current default behaviour (generic SFT adapter from current `train_lora.py`) on a fixed 100-question test set per workspace. This becomes the measured baseline.
2. After each workstream completes, re-run on the same test set. Improvement delta is reported in MLflow.
3. Acceptance for any workstream requires non-regression on previously achieved metrics (regression > 5pp triggers rollback).

Qualitative:
- Non-ML user can run end-to-end training without CLI
- Adapter version v2 retains v1 quality after continual training
- Embedder fine-tuning measurably improves NDCG@10 on domain corpus

## 10. Alignment with existing project vision

This spec is **subordinate** to `2026-05-22-project-vision-design.md`. Specifically:

- The vision document (sec 4.1) names **LoRA per-tenant fine-tuning** as main moat
- Vision Phase 2 (months 3-4) lists "LoRA Auto-Train UI" as a workstream
- Pack B is the **detailed technical specification** of how that workstream (and its prerequisites) are built

Pack B does NOT supersede vision; it implements one branch of it. Other vision components (Compliance Mode, GigaChat/YandexGPT integration, Docling showcase) remain separate workstreams.

## 11. Open questions

None at this design stage. All technical decisions are settled. Implementation details (hyperparameter defaults, file paths) will be refined during the writing-plans phase.
