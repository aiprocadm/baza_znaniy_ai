# ML Strengthening — Pack B+ Design

**Date:** 2026-05-25
**Author scope:** technical design for ML capability uplift in KB.AI
**Status:** Design document. Subordinate to `2026-05-22-project-vision-design.md` (LoRA as main moat).
**Decision context:** Technical analysis of existing ML stack identified 7 critical gaps and 11 secondary improvements. Pack B+ bundles the 9 critical-gap workstreams (W1-W10) plus 4 commercially-valuable items from former Pack C (W11-W14: multi-adapter composition, structured outputs, model registry, knowledge distillation) into a coherent ML platform uplift. Items reserved for hypothetical future expansion (TIES/DARE merging, QAT, multi-node DeepSpeed) remain out of scope.

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

## 3. Pack B+ scope — 14 workstreams

Pack B+ implements all 7 critical gaps (W1, W2, W3, W4, W5, W6, W7), 2 supporting items (W8 MLflow, W10 Golden-set regression), the Auto-Train UI orchestrator (W9), plus 4 commercially-valuable items lifted from former Pack C (W11 multi-adapter composition, W12 structured outputs, W13 model registry, W14 knowledge distillation). Pack A (gaps G1, G4, G7 + regression test) is **strict subset** of Pack B+.

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

---

### Workstream 11: Multi-adapter composition (Pack C item C1)

**Goal:** activate 2-3 LoRA adapters simultaneously with independent scale factors, allowing combination of orthogonal skills (e.g. "company glossary" + "concise response style" + "legal terminology").

**Architectural challenge:** `llama.cpp` supports hot-swap of a single adapter via `set_adapter()` but does NOT natively support multi-adapter composition with weights. Two paths:

1. **Transformers/PEFT path** (training + experimentation):
   - PEFT's `model.add_weighted_adapter([name_a, name_b], weights=[1.0, 0.6], adapter_name="composed")` produces a merged in-memory model
   - Useful during training and CPU inference
   - Slower than llama.cpp at runtime

2. **Pre-merged GGUF path** (production inference):
   - Compose adapters once via PEFT, then export the merged result as a single new adapter
   - Convert to GGUF and load in llama.cpp normally
   - Fast inference, but composition is "frozen" until next merge

**Changes:**
- New file: `app/llm/transformers_provider.py` — PEFT-based provider supporting `load_adapters(names, weights)`
- Extension to `app/llm/lora_runtime.py`:
  - `AdapterSlot` dataclass with `name`, `scale`, `path`
  - `_ACTIVE_ADAPTERS: list[AdapterSlot]` replaces single `_ACTIVE_ADAPTER`
  - `compose_adapters(slots)` — assembles via PEFT, optionally exports merged GGUF for llama.cpp
- New file: `scripts/compose_adapters.py` — CLI for pre-merging adapters
- New API: `POST /api/v1/lora/compose` with body `{"slots": [{"name": "glossary", "scale": 1.0}, {"name": "concise", "scale": 0.6}]}`
- UI: in Auto-Train UI (W9), add "Compose adapters" step — multi-select adapters with scale sliders

**Acceptance:**
- Composing 2 adapters (glossary + style) produces a model that exhibits behaviour of both, verified via RAGAS faithfulness (must preserve glossary's domain accuracy) AND style metrics (response length distribution shifted toward concise)
- Pre-merged GGUF export round-trips through llama.cpp with no functional regression

---

### Workstream 12: Structured outputs (Pack C item C2)

**Goal:** force LLM to produce output strictly matching a user-specified JSON schema or grammar — critical for "AI fills our forms" use cases (Н-1 reports, SOUT cards, legal extracts).

**Two implementation paths:**

1. **`outlines` library (default, universal):**
   - `outlines.generate.json(model, pydantic_or_schema)` → guaranteed valid JSON
   - Works with both `transformers` and `llama.cpp` backends
   - More flexible (regex, choice, format constraints)

2. **`llama.cpp grammar.gbnf` (fast, native):**
   - Define grammar file in GBNF (GGML BNF) syntax
   - Pass as `grammar` parameter to `Llama.create_completion()`
   - Lower overhead, llama.cpp-only

**Changes:**
- New file: `app/services/structured_outputs.py` — abstraction over both backends
- New API: `POST /api/kb/ask/structured` with body:
  ```json
  {
    "question": "Составь акт Н-1 по этому инциденту",
    "context_filter": {"category": "incidents"},
    "schema": {
      "type": "object",
      "properties": {
        "actNumber": {"type": "string"},
        "victim": {"type": "string"},
        "date": {"type": "string", "format": "date"},
        "circumstances": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "object", "properties": {"doc": {"type": "string"}, "page": {"type": "integer"}}}}
      },
      "required": ["actNumber", "victim", "date", "citations"]
    }
  }
  ```
- Response: validated JSON matching the schema
- New file: `data/schemas/` — repository of pre-built schemas for common domains (incident reports, contract extracts, SOP summaries)
- UI: "Schemas" tab in Operations Console — manage / preview / test schemas

**Integration with RAG:** structured generation runs on the RAG-aware tuned model from W3 — the model uses retrieved context to fill the JSON, with citations as a required field.

**Acceptance:**
- 95%+ requests produce JSON that validates against the schema
- Citation accuracy (faithfulness) preserved vs. free-text generation (RAGAS check)
- Both `outlines` and `grammar.gbnf` paths work for the same schema

---

### Workstream 13: Model registry & hot-switch (Pack C item C3)

**Goal:** support multiple base models on the same server with hot-switch between them, without process restart.

**Current state:** `LLM_MODEL_PATH=./models/model.gguf` hardcoded in settings; switch requires editing env + restart.

**Changes:**

1. **Model registry structure** — analogous to existing LoRA registry:
   ```
   var/models/
     saiga-llama3-8b/
       manifest.json           # {"name", "format": "gguf", "base_arch": "llama", "ctx": 8192}
       model.gguf
     qwen2.5-7b-instruct/
       manifest.json
       model.gguf
     tinyllama-1.1b/
       manifest.json
       model.gguf
   ```

2. **`app/llm/model_registry.py`** — new module:
   - `list_models()` → list[ModelInfo]
   - `activate_model(name)` → reload llama.cpp with new model_path
   - Handles graceful shutdown of current model + reload + restore active adapter

3. **API endpoints:**
   - `GET /api/v1/models` — list available
   - `POST /api/v1/models/{name}/activate` — hot-switch (blocks 10-30s during reload)
   - `GET /api/v1/models/active` — current active model

4. **Per-workspace default:** workspace can pin a preferred model (extends future workspace model from Pack B vision); falls back to global default if unset.

5. **UI:** in Operations Console, replace static "model" display chip with dropdown selector showing all registered models, with active one highlighted.

**Adapter compatibility:** during switch, the active LoRA adapter is unloaded if `adapter.base != new_model.name`. Auto-Train UI must validate base-model compatibility before allowing adapter activation.

**Acceptance:**
- Server reports 3 registered models via `GET /api/v1/models`
- Switching from Saiga to Qwen takes <30s and the next `/api/kb/ask` request uses the new model
- LoRA adapter trained on Saiga is rejected with HTTP 409 when active model is Qwen

---

### Workstream 14: Knowledge distillation (Pack C item C5)

**Goal:** distill a large teacher model (DeepSeek-V3 or Llama-3-70B) into a smaller student (Saiga-3B or Phi-3-mini) on the customer's domain corpus, producing a fast on-premise-friendly model with teacher-level quality.

**Two distillation modes:**

1. **Hard distillation** (default, supported by all teacher APIs):
   - Generate large-scale Q&A dataset from corpus using teacher (extends W1 scale: 50k+ examples)
   - Train student via SFT on (input, teacher_output) pairs
   - Loss: standard CrossEntropy

2. **Soft distillation** (requires teacher with logits access — only some local teachers):
   - Capture teacher's full logit distribution per token
   - Train student with combined loss: `α * KL(student_logits, teacher_logits) + (1-α) * CrossEntropy(student, target)`
   - Higher quality, requires teacher model running locally (not API)

**Default implementation:** hard distillation, because public APIs (DeepSeek, Groq, OpenAI, OpenRouter) do not expose token logits. Soft distillation enabled when teacher is local llama.cpp model.

**Changes:**

1. **`scripts/distill_dataset.py`** — large-scale extension of W1's synthetic generator:
   - Default: 50,000 Q&A pairs (vs. W1's typical 1000)
   - Cost estimate: $25-50 via DeepSeek-V3 (~$0.0005-0.001 per example)
   - Diversity boost: aggressive paraphrase modes, multi-hop questions, edge cases

2. **`scripts/train_distillation.py`** — student training:
   - Base model: configurable (Saiga-3B, Phi-3-mini, TinyLlama)
   - Loss mode: `--mode hard` (default) or `--mode soft` (requires `--teacher-path`)
   - LoRA OR full fine-tune (config flag)
   - MLflow integration (W8)

3. **Quality validation:**
   - Student must achieve ≥85% of teacher's RAGAS faithfulness on held-out test
   - Latency improvement: student inference 5-10x faster than teacher (target metric)

4. **Integration with model registry (W13):**
   - Distilled student lands in `var/models/<workspace>-distilled-saiga-3b/`
   - Becomes available for activation alongside base models
   - Manifest records `teacher` field for provenance

**Practical constraint:** distillation works best for narrow domains (single workspace's corpus). Cross-workspace distillation is out of scope.

**Acceptance:**
- Distillation pipeline: 50k Q&A generation completes in <4h for ~$30
- Student Saiga-3B trained on distilled data achieves RAGAS faithfulness ≥ 0.85 * teacher score
- Student inference latency on CPU is <3s per question (vs. teacher >30s or API ~1-2s + network)

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
      ├──> [W11: compose_adapters.py] ──> merged composite adapter
      │       │
      │       └──> [W13: model_registry / hot-switch] ──> active model + adapter slots
      │
      ├──> [W12: structured_outputs.py] ──> /ask/structured endpoint
      │
      ├──> [W14: distill_dataset.py (50k scale)] ──> distilled_qa.jsonl
      │       │
      │       └──> [W14: train_distillation.py] ──> student model in registry
      │
      └──> [W9: Auto-Train UI] ──> orchestrates W1-W7, W11-W14 visually
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
9. **W13 (model registry)** — seventh; needed before W11 multi-model composition; can also run in parallel after W2
10. **W11 (multi-adapter composition)** — eighth; depends on W4 (DPO adapters available) and W13 (registry to source models)
11. **W12 (structured outputs)** — ninth; depends on W3 (RAG-aware model) for citations field
12. **W14 (knowledge distillation)** — tenth; depends on W1 (data generator), W5 (RAGAS for student validation), W13 (registry for student deployment)
13. **W9 (Auto-Train UI)** — last UI-side; wraps W1-W7 + W11-W14; depends on all functional
14. **W10 (regression)** — added in parallel with W5

## 6. Files to be created or modified

**New files (W1-W10):**
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

**New files (W11-W14 from Pack C additions):**
- `app/llm/transformers_provider.py` — PEFT-based provider for multi-adapter composition (W11)
- `scripts/compose_adapters.py` — CLI for pre-merging adapter weights (W11)
- `app/services/structured_outputs.py` — abstraction over outlines + grammar.gbnf (W12)
- `data/schemas/` — pre-built JSON schemas for common domains (W12)
- `app/llm/model_registry.py` — model registry + hot-switch logic (W13)
- `scripts/distill_dataset.py` — large-scale Q&A generation for distillation (W14)
- `scripts/train_distillation.py` — student model trainer with hard/soft modes (W14)
- `var/models/` — directory for registered base models (W13, W14 student outputs)

**Modified files:**
- `scripts/train_lora.py` — auto chat template, architecture-aware target modules, `--resume-from-adapter`, `--ewc-lambda`, `--prompt-mode {generic, rag}`, MLflow integration
- `scripts/eval_lora.py` — MLflow integration; mark EM/ROUGE as deprecated
- `app/services/kb_embeddings.py` — `local` backend
- `app/services/lora_manager.py` — regression check before activation; multi-slot adapter activation (W11)
- `app/services/reindex_service.py` — `--embedder-changed` flag
- `app/api/kb_mvp.py` — `/messages/{id}/feedback`, `/feedback/export`, `/ask/structured` (W12)
- `app/api/v1/routes_lora.py` — `/lora/compose` endpoint (W11)
- `app/llm/lora_runtime.py` — `AdapterSlot` dataclass, `_ACTIVE_ADAPTERS` list (W11)
- `app/llm/llama_cpp_provider.py` — graceful close + reload for hot-switch (W13)
- `app/core/config.py` — replace single `LLM_MODEL_PATH` with registry-aware resolution (W13)
- `requirements-train.txt` — add `trl`, `ragas`, `mlflow`, `sentence-transformers[train]`, `accelerate`, `outlines`, `mergekit` (W11 alternative), `peft[train]`
- `requirements-runtime.txt` — add `mlflow-skinny` (for runtime logging only), `outlines` (W12)

## 7. Out of scope (future hypothetical packs)

Explicitly excluded from Pack B+ — to be addressed in separate future specs only if customer demand emerges:

- **TIES / DARE adapter merging** (`mergekit`) — internal optimisation, not customer-visible. May be revisited if multi-adapter composition (W11) hits performance limits.
- **Quantization-Aware Training (QAT)** — +2-5% quality after Q4_K_M quantisation. Deep ML expertise required, low customer visibility.
- **Distributed training beyond FSDP** (DeepSpeed ZeRO-3, multi-node) — most customers lack multi-node infrastructure. Pack B's FSDP (single-node multi-GPU) covers the realistic 99% case.
- **Cross-workspace knowledge distillation** — Pack B+ supports per-workspace distillation (W14); cross-workspace transfer would require additional privacy guarantees and is out of scope.
- **Reward modelling for RLHF** (full PPO/RLHF loop) — DPO (W4) is the modern replacement and is sufficient for production.
- **Multi-modal extension** (images, audio) — KB.AI is text-focused; multi-modal would be a different product.

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
| Multi-adapter composition (W11) — llama.cpp limitations on simultaneous adapters | Pre-merge via PEFT + GGUF export (frozen composition for production); transformers backend for interactive experimentation |
| Structured outputs (W12) — schema-constrained generation degrades quality | A/B test with/without schema on golden set; allow "soft" mode (validate but not enforce) for sensitive prompts |
| Model hot-switch (W13) — 10-30s blocking reload disrupts users | Display reload progress in UI; queue incoming requests with timeout; document expected downtime per switch |
| Knowledge distillation (W14) — student fails to match teacher quality | Validate via RAGAS BEFORE adding to registry; document achievable quality ceilings per student size (3B → 85%, 1.5B → 70%) |
| Distillation API costs (W14) — unexpected spend | Budget cap in CLI (`--max-budget-usd`); cost preview before run; refuse if exceeds |

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
