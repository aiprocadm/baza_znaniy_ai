# Hybrid Eval Corpus — PR2: Synthetic Public Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit a reproducible synthetic public eval corpus (~9 RU docs, 300+ chunks), a public golden set (auto breadth + curated hard cases), and frozen bge-m3 embeddings — and retire the private contract golden from git, per the accepted spec §13 decision.

**Architecture:** The committed artifact is *text + vectors*, never a generator run: synthetic `.md` fixtures under `data/eval/corpus_public/`, `golden_public.jsonl` with composite `"<filename>:<idx>"` keys (PR1 scheme), and a pickle-free `frozen_bge-m3.npz` + `keys.json` pair that PR3's CI gate will rank with pure numpy. Two small scripts (`build_public_corpus.py` ingest/author, `build_frozen_embeddings.py`) are DI-tested without real models; the real bge-m3/GGUF runs are one-time operator steps.

**Tech Stack:** Python 3.13 (`py -3.13`), pytest + `tests/stubs/`, numpy, the in-process `st` embedder (BAAI/bge-m3) and keyless GGUF provider (Qwen2.5-3B) already on `main`.

**Source spec:** [`docs/superpowers/specs/2026-06-06-hybrid-eval-corpus-design.md`](../specs/2026-06-06-hybrid-eval-corpus-design.md) §6, §13; roadmap section "PR2" in [`2026-06-06-hybrid-eval-corpus.md`](2026-06-06-hybrid-eval-corpus.md).

**Conventions:** run tests with `py -3.13 -m pytest ... --ignore=backend`; confirm pass/fail by exit code (`$LASTEXITCODE`), not piped output. Conventional Commits. The default embedder is now ST (keyless stack #585) — unit tests must inject a fake embedder or set `KB_EMBEDDINGS_BACKEND=hash` to stay deterministic.

---

## Verified in-tree interfaces (2026-06-10, main @ 9e8e8fe)

- `KnowledgeBaseStore(db_path, embedder=...)`; `add_document(title, text=None, *, pages=None, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP, source="text", filename=None, mime_type=None) -> Document`; `list_documents() -> List[Document]`; `delete_document(doc_id) -> bool`; module-level `get_store()` honours `KB_MVP_DB_PATH`.
- `app.eval.adapter`: `_build_key_map(store)`, `build_global_id_key_map(store)`, `compute_signature(store)`, `EvalHit(chunk_key, text, title)`.
- `app.eval.dataset`: `GoldenItem(question, relevant_chunks: tuple[str,...], reference_answer, expect_refusal, source)`, `load_golden`, `save_golden`, `write_signature`, `read_signature`.
- `app.services.kb_embeddings.get_embedder(env=None)` (cached; `reset_embedder()` clears), `SentenceTransformerEmbedder(model_name=, e5_prefix_enabled=, model=)` with `embed()` / `embed_query()` / `dimension`.
- `scripts/eval_rag.py generate --out … --limit N --yes`; `run --golden` **defaults to `data/eval/golden_curated.jsonl`** (Task 7 repoints it).
- `tests/conftest.py::_PROTECTED_FIXTURES` write-protects `golden_curated.*` (Task 7 swaps entries).
- Local models: `models/qwen2.5-3b-instruct-q4_k_m.gguf` present (2 GB). bge-m3 downloads to HF cache on first use (~2.2 GB) — not cached yet on this machine.

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `scripts/build_public_corpus.py` | `ingest` *.md fixtures into the env-selected store; `author` drafts ONE new doc via `select_provider` (extension path) | **Create** |
| `tests/scripts/test_build_public_corpus.py` | DI unit tests for both modes (fake store path / fake provider) | **Create** |
| `data/eval/corpus_public/*.md` | 9 synthetic RU documents (committed text fixtures) | **Create** |
| `data/eval/golden_public.jsonl` + `.sig.json` | auto breadth + curated hard cases, composite keys | **Create** |
| `scripts/build_frozen_embeddings.py` | embed passages+queries with the active embedder → `.npz` (float32 only) + `.keys.json` | **Create** |
| `tests/scripts/test_build_frozen_embeddings.py` | DI unit tests (fake embedder, tmp store/golden) | **Create** |
| `data/eval/corpus_public/frozen_bge-m3.npz` + `frozen_bge-m3.keys.json` | committed frozen vectors + string sidecar | **Create** |
| `scripts/eval_rag.py:147` | `run --golden` default → `golden_public.jsonl` | Modify |
| `tests/conftest.py:58-62` | protect `golden_public.*` instead of `golden_curated.*` | Modify |
| `scripts/build_curated_golden.py`, `data/eval/golden_curated.jsonl` + `.sig.json` | retire to private (copy under `var/data/eval/private/`, then `git rm`) | **Delete** |
| `tests/test_golden_curated.py` → `tests/test_golden_public.py` | validate the committed public golden | Rename/Modify |
| `tests/test_eval_generation.py:48,59,62` | PR1 carry-over: int keys → string keys | Modify |

---

## Task 0: Branch + environment sanity

**Files:** none (verification only)

- [ ] **Step 1: Branch off fresh main**

```powershell
git checkout main; git pull --ff-only
git checkout -b feat/eval-public-corpus
```

- [ ] **Step 2: Verify the model stack imports and the GGUF is present**

```powershell
py -3.13 -c "import llama_cpp, sentence_transformers, numpy; print('ml ok')"
Test-Path models/qwen2.5-3b-instruct-q4_k_m.gguf   # → True
```
Expected: `ml ok` and `True`. If `llama_cpp` fails to import, STOP — the golden auto-generation (Task 4) depends on it.

- [ ] **Step 3: Confirm the eval harness baseline is green**

Run: `py -3.13 -m pytest tests/test_eval_dataset.py tests/test_eval_adapter.py tests/test_eval_retrieval.py tests/test_eval_cli.py -q; $LASTEXITCODE`
Expected: exit `0`.

---

## Task 1: `build_public_corpus.py` — `ingest` mode (TDD)

**Files:**
- Create: `scripts/build_public_corpus.py`
- Test: `tests/scripts/test_build_public_corpus.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_build_public_corpus.py`:
```python
"""Unit tests for the public-corpus build script (DI — no real models)."""

from __future__ import annotations

from pathlib import Path

from app.services.kb_store import KnowledgeBaseStore
from scripts.build_public_corpus import derive_title, ingest_corpus


class _FakeEmbedder:
    name = "fake"
    dimension = 4

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


def test_derive_title_prefers_first_heading() -> None:
    assert derive_title("# Договор оказания услуг\n\nТекст", "x.md") == "Договор оказания услуг"
    assert derive_title("без заголовка", "contract_services.md") == "contract_services"


def _write(corpus: Path, name: str, body: str) -> None:
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / name).write_text(body, encoding="utf-8")


def test_ingest_adds_every_md_with_filename(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    _write(corpus, "a.md", "# Док А\n\n" + "текст про оплату. " * 30)
    _write(corpus, "b.md", "# Док Б\n\n" + "текст про сроки. " * 30)
    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"), embedder=_FakeEmbedder())

    count = ingest_corpus(store, corpus)

    docs = store.list_documents()
    assert count == 2 and len(docs) == 2
    assert sorted(d.filename for d in docs) == ["a.md", "b.md"]
    assert sorted(d.title for d in docs) == ["Док А", "Док Б"]


def test_ingest_is_idempotent_replaces_same_filename(tmp_path) -> None:
    corpus = tmp_path / "corpus"
    _write(corpus, "a.md", "# Версия 1\n\n" + "старый текст. " * 30)
    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"), embedder=_FakeEmbedder())
    ingest_corpus(store, corpus)

    _write(corpus, "a.md", "# Версия 2\n\n" + "новый текст. " * 30)
    count = ingest_corpus(store, corpus)

    docs = store.list_documents()
    assert count == 1 and len(docs) == 1
    assert docs[0].title == "Версия 2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/scripts/test_build_public_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_public_corpus'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/build_public_corpus.py`:
```python
"""Build / maintain the synthetic public eval corpus.

The committed artifact is the *text* under ``data/eval/corpus_public/*.md`` —
never a generator run (spec 2026-06-06 §6: determinism). Two modes:

``ingest``
    Load every ``*.md`` fixture into the store selected by the environment
    (``KB_MVP_DB_PATH`` — point it at the public store). Idempotent: a document
    whose ``filename`` already exists is deleted and re-added, so re-running
    after editing a fixture cannot duplicate chunks.

``author``
    Draft ONE new document with the configured LLM (``select_provider`` — the
    keyless local GGUF by default) into the corpus directory for human review.
    Extension path only; the initial nine documents were teacher-authored and
    committed as reviewed text.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.services.kb_store import KnowledgeBaseStore, get_store

LOGGER = logging.getLogger(__name__)

CORPUS_DIR = Path("data/eval/corpus_public")

_AUTHOR_PROMPT = (
    "Ты — опытный корпоративный юрист и методолог. Напиши реалистичный "
    "внутренний документ российской компании в формате Markdown.\n"
    "Тип документа: {doc_type}\nТема: {topic}\n"
    "Требования: 3000-5000 слов; нумерованные разделы; конкретные выдуманные "
    "реквизиты, суммы, сроки и должности (внутренне непротиворечивые); без "
    "реальных персональных данных и реальных компаний. Начни с заголовка "
    "первого уровня '# '."
)


def derive_title(text: str, filename: str) -> str:
    """First markdown H1 wins; otherwise the filename stem."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return Path(filename).stem


def ingest_corpus(store: KnowledgeBaseStore, corpus_dir: Path) -> int:
    """Load every ``*.md`` in *corpus_dir* into *store*. Returns files ingested."""
    existing = {d.filename: d.id for d in store.list_documents() if d.filename}
    count = 0
    for md in sorted(corpus_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8")
        if md.name in existing:
            store.delete_document(existing[md.name])
        store.add_document(
            derive_title(text, md.name), text=text, source="text", filename=md.name
        )
        count += 1
        LOGGER.info("ingested %s", md.name)
    return count


def author_document(name: str, doc_type: str, topic: str, *, provider=None) -> Path:
    """Draft one new corpus document for human review (extension path)."""
    if provider is None:
        from app.services.kb_llm import select_provider

        provider = select_provider()
    if provider is None:
        raise SystemExit("No LLM provider available (need the keyless GGUF or a key)")
    prompt = _AUTHOR_PROMPT.format(doc_type=doc_type, topic=topic)
    response = provider.generate(prompt, max_tokens=8192)
    out = CORPUS_DIR / f"{name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(response.text.strip() + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="load corpus_public/*.md into the env-selected store")
    ing.add_argument("--corpus-dir", default=str(CORPUS_DIR))

    auth = sub.add_parser("author", help="draft ONE new document via the configured LLM")
    auth.add_argument("name", help="output filename stem, e.g. policy_travel")
    auth.add_argument("--doc-type", required=True)
    auth.add_argument("--topic", required=True)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        n = ingest_corpus(get_store(), Path(args.corpus_dir))
        print(f"OK: ingested {n} document(s)")
    else:
        out = author_document(args.name, args.doc_type, args.topic)
        print(f"Draft written to {out} — review before committing")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.13 -m pytest tests/scripts/test_build_public_corpus.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint + commit**

```powershell
py -3.13 -m ruff check scripts/build_public_corpus.py tests/scripts/test_build_public_corpus.py
py -3.13 -m black scripts/build_public_corpus.py tests/scripts/test_build_public_corpus.py
git add scripts/build_public_corpus.py tests/scripts/test_build_public_corpus.py
git commit -m "feat(eval): public-corpus build script (ingest + author modes)"
```

---

## Task 2: Author the nine synthetic documents

**Files:**
- Create: `data/eval/corpus_public/<name>.md` × 9

**Method:** teacher-LLM authoring (this session's model is the teacher; spec §6 allows "any configured teacher-LLM" and commits only the output text). Dispatch parallel subagents, one per document, with the spec below; review each against the acceptance checklist before committing.

- [ ] **Step 1: Author the documents (parallel subagents, 3 batches of 3)**

Document specs — every document must use ONLY fictional companies/people, contain concrete internally-consistent numbers (суммы, сроки, пороги, должности) answerable by exact-fact questions, be 25 000–35 000 characters of Russian Markdown with numbered sections, and start with `# <title>`:

| File | Type | Content requirements |
|------|------|---------------------|
| `contract_services.md` | Договор оказания услуг | ИТ-сопровождение между ООО «Вектор Плюс» (заказчик) и ООО «ТехСервис» (исполнитель): абонентская плата, тарифы сверх лимита, SLA (время реакции/восстановления по приоритетам), штрафы, срок действия, порядок расторжения, реквизиты |
| `contract_supply.md` | Договор поставки | Поставка офисного оборудования: спецификация с ценами, сроки поставки/замены брака, гарантия, неустойка за просрочку, условия оплаты (аванс/постоплата) |
| `nda.md` | Соглашение о неразглашении | Взаимное NDA: определение конфиденциальной информации, исключения, срок действия обязательств после расторжения, штраф за разглашение, порядок возврата носителей |
| `reglament_infosec.md` | Регламент | Информационная безопасность: парольная политика (длина/ротация), уровни доступа, работа с USB, действия при инциденте и сроки уведомления, ответственность |
| `reglament_ot.md` | Регламент | Охрана труда: инструктажи (виды и периодичность), медосмотры, выдача СИЗ, действия при несчастном случае, ответственные роли |
| `procedure_onboarding.md` | Процедура | Адаптация новых сотрудников: этапы по дням (1-й день, первая неделя, испытательный срок), роли (наставник, HR, руководитель), чек-листы, критерии прохождения ИС |
| `procedure_procurement.md` | Процедура | Закупки: пороги сумм и уровни согласования, сроки рассмотрения заявок, количество требуемых КП, критерии выбора поставщика, документооборот |
| `policy_remote_work.md` | Положение | Удалённая работа: кто имеет право, порядок оформления, требования к рабочему месту и ИБ, компенсации, контроль рабочего времени, отзыв с удалёнки |
| `npa_pdn_excerpt.md` | NPA-стиль | Положение об обработке персональных данных (синтетическое, в духе 152-ФЗ): категории ПДн, правовые основания, сроки хранения, права субъектов, сроки ответа на запросы |

Subagent prompt template (fill the row values):
```
Напиши синтетический документ для публичного eval-корпуса RAG-системы.
Файл: data/eval/corpus_public/<file>. Тип: <type>. Требования: <content requirements>.
Объём 25000-35000 символов, русский Markdown, нумерованные разделы, начни с '# <заголовок>'.
Только вымышленные компании/имена/реквизиты. Числа и сроки конкретные и внутренне
непротиворечивые (на них будут задаваться вопросы с точными ответами). Не вставляй
комментарии вне документа. Запиши результат в файл инструментом Write.
```

- [ ] **Step 2: Review every document against the acceptance checklist**

For each file verify, fixing by editing the file directly:
- starts with `# `; length 25 000–35 000 chars (use `(Get-Content $f -Raw).Length` — `(Get-Item $f).Length` is bytes, ~2× for Cyrillic UTF-8);
- no real company/person names; numbers internally consistent (e.g. the SLA table matches the penalties section);
- facts are *specific* enough for exact-answer Q&A (sums, days, thresholds);
- no answer-leakage style ("Ответ:", Q&A-форматирование) — these are documents, not quizzes.

- [ ] **Step 3: Commit the fixtures**

```powershell
git add data/eval/corpus_public/
git commit -m "feat(eval): synthetic public RU corpus (9 docs, teacher-authored)"
```

---

## Task 3: Ingest into the public store (operator step)

**Files:** none committed (`var/data/kb_public.sqlite` is gitignored)

- [ ] **Step 1: Ingest under the real bge-m3 embedder**

```powershell
$env:KB_MVP_DB_PATH="var/data/kb_public.sqlite"
$env:KB_EMBEDDINGS_BACKEND="st"; $env:ST_EMBED_MODEL="BAAI/bge-m3"
py -3.13 -m scripts.build_public_corpus ingest
```
Expected: `OK: ingested 9 document(s)`. First run downloads bge-m3 (~2.2 GB) into the HF cache. Embedding ~300+ chunks on CPU takes minutes.

- [ ] **Step 2: Verify the corpus signature**

```powershell
py -3.13 -c "import os; os.environ.setdefault('KB_MVP_DB_PATH','var/data/kb_public.sqlite'); os.environ.setdefault('KB_EMBEDDINGS_BACKEND','st'); os.environ.setdefault('ST_EMBED_MODEL','BAAI/bge-m3'); from app.services.kb_store import get_store; from app.eval.adapter import compute_signature; s=compute_signature(get_store()); print(s.to_dict())"
```
Expected: `embedder_name` containing `st`, `dim` 1024, `doc_count` 9, chunk count ≥ 250. If chunk count < 250, extend the shortest documents (Task 2 Step 2) and re-run Step 1.

---

## Task 4: Auto golden breadth via the keyless GGUF (operator step)

**Files:**
- Create: `data/eval/golden_public.jsonl` (first version, auto-only)

- [ ] **Step 1: Generate with the local stack (no keys needed — GGUF is the default)**

```powershell
$env:KB_MVP_DB_PATH="var/data/kb_public.sqlite"
$env:KB_EMBEDDINGS_BACKEND="st"; $env:ST_EMBED_MODEL="BAAI/bge-m3"
py -3.13 -m scripts.eval_rag generate --out data/eval/golden_public.jsonl --limit 60 --yes
```
Expected: progress over ≤60 chunks; minutes-per-chunk on CPU (3B GGUF + self-consistency check). `--limit 60` bounds the run to roughly 1–2 h; raise later runs to extend breadth (spec target is breadth over the corpus, the curated cases carry the hard signal).

- [ ] **Step 2: Spot-check quality**

Read 10 random lines of `data/eval/golden_public.jsonl`; verify questions are non-trivial, answers match the document facts, every `relevant_chunks` key is composite (`":" in key`). Delete clearly trivial/leaky items (the review gate from spec §6).

- [ ] **Step 3: Commit the auto layer**

```powershell
git add data/eval/golden_public.jsonl
git commit -m "feat(eval): auto golden breadth for the public corpus (local GGUF)"
```

---

## Task 5: Curated hard cases + signature sidecar

**Files:**
- Modify: `data/eval/golden_public.jsonl` (append curated items)
- Create: `data/eval/golden_public.sig.json`

- [ ] **Step 1: Map chunk keys for the facts you will target**

```powershell
$env:KB_MVP_DB_PATH="var/data/kb_public.sqlite"; $env:KB_EMBEDDINGS_BACKEND="st"; $env:ST_EMBED_MODEL="BAAI/bge-m3"
py -3.13 -c "from app.services.kb_store import get_store; from app.eval.adapter import _build_key_map; km=_build_key_map(get_store()); from collections import Counter; print(Counter(k.split(':')[0] for k in km.values()))"
```
Then, for each curated question, locate its supporting chunk(s) by querying the store (`store.search("<вопрос>", top_k=5)` and reading the hit's `document_id`/`chunk_index`), or by grepping the fixture text and counting chunk boundaries (chunk stride = 900 chars with 140 overlap).

- [ ] **Step 2: Author ≥15 curated items and append**

Write a throwaway local script (not committed) that builds `GoldenItem` objects and appends via `save_golden` — composition requirements:
- ≥3 `expect_refusal=True` items (questions whose answers are NOT in the corpus, e.g. «Какова зарплата генерального директора?»), `relevant_chunks=()`;
- ≥3 multi-hop items (`relevant_chunks` spanning 2+ documents, e.g. сопоставить порог закупки из `procedure_procurement.md` с суммой договора из `contract_supply.md`);
- ≥3 paraphrase items (same fact as an auto item, asked in different words);
- remainder: exact-fact questions on tables/numbers (SLA, штрафы, сроки);
- all with `source="curated"`.

Append pattern:
```python
from pathlib import Path
from app.eval.dataset import GoldenItem, load_golden, save_golden

GOLDEN = Path("data/eval/golden_public.jsonl")
items = load_golden(GOLDEN)
items += [
    GoldenItem("…вопрос…", ("contract_services.md:4",), "…ответ…", source="curated"),
    GoldenItem("…вопрос вне корпуса…", (), "", expect_refusal=True, source="curated"),
    # …
]
save_golden(GOLDEN, items)
```

- [ ] **Step 3: Write the signature sidecar**

```powershell
py -3.13 -c "from pathlib import Path; from app.services.kb_store import get_store; from app.eval.adapter import compute_signature; from app.eval.dataset import write_signature; write_signature(Path('data/eval/golden_public.jsonl'), compute_signature(get_store())); print('sig written')"
```
(Same env as Step 1.) Expected: `data/eval/golden_public.sig.json` appears with `embedder_name` = st-family, `dim` 1024.

- [ ] **Step 4: Validate every key resolves against the store**

```powershell
py -3.13 -c "from pathlib import Path; from app.services.kb_store import get_store; from app.eval.adapter import _build_key_map; from app.eval.dataset import load_golden; km=set(_build_key_map(get_store()).values()); bad=[k for it in load_golden(Path('data/eval/golden_public.jsonl')) for k in it.relevant_chunks if k not in km]; print('bad keys:', bad)"
```
Expected: `bad keys: []`. Fix any stragglers before committing.

- [ ] **Step 5: Commit**

```powershell
git add data/eval/golden_public.jsonl data/eval/golden_public.sig.json
git commit -m "feat(eval): curated hard cases + signature for the public golden"
```

---

## Task 6: `build_frozen_embeddings.py` (TDD) + real freeze

**Files:**
- Create: `scripts/build_frozen_embeddings.py`
- Test: `tests/scripts/test_build_frozen_embeddings.py`
- Create (operator run): `data/eval/corpus_public/frozen_bge-m3.npz`, `data/eval/corpus_public/frozen_bge-m3.keys.json`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_build_frozen_embeddings.py`:
```python
"""Unit tests for the frozen-embeddings builder (DI — no real model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.eval.dataset import GoldenItem, save_golden
from app.services.kb_store import KnowledgeBaseStore
from scripts.build_frozen_embeddings import build_frozen, write_frozen


class _FakeEmbedder:
    name = "fake-st"
    dimension = 4

    def embed(self, text: str) -> list[float]:
        seed = float(len(text) % 7 + 1)
        return [seed, 0.0, 0.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, float(len(text) % 5 + 1), 0.0, 0.0]


def _store_with_chunks(tmp_path) -> KnowledgeBaseStore:
    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"), embedder=_FakeEmbedder())
    store.add_document("Док", text="первый чанк. " * 40, filename="doc.md")
    return store


def test_build_frozen_shapes_keys_and_normalization(tmp_path) -> None:
    store = _store_with_chunks(tmp_path)
    golden = tmp_path / "golden.jsonl"
    save_golden(golden, [GoldenItem("вопрос один?", ("doc.md:0",), "a")])

    frozen = build_frozen(store, _FakeEmbedder(), golden)

    assert frozen.passage_vecs.dtype == np.float32
    assert frozen.passage_vecs.shape[0] == len(frozen.passage_keys) >= 1
    assert all(":" in k for k in frozen.passage_keys)
    assert frozen.query_vecs.shape == (1, 4)
    assert frozen.query_texts == ["вопрос один?"]
    norms = np.linalg.norm(frozen.passage_vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)  # L2-normalized


def test_write_frozen_loads_without_object_arrays(tmp_path) -> None:
    store = _store_with_chunks(tmp_path)
    golden = tmp_path / "golden.jsonl"
    save_golden(golden, [GoldenItem("вопрос?", ("doc.md:0",), "a")])
    frozen = build_frozen(store, _FakeEmbedder(), golden)

    npz, keys = write_frozen(frozen, tmp_path / "out", "fake-st")

    loaded = np.load(npz)  # default loader (no object arrays) — must not raise
    assert set(loaded.files) == {"passage_vecs", "query_vecs"}
    meta = json.loads(Path(keys).read_text(encoding="utf-8"))
    assert meta["passage_keys"] == list(frozen.passage_keys)
    assert meta["query_texts"] == ["вопрос?"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.13 -m pytest tests/scripts/test_build_frozen_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_frozen_embeddings'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/build_frozen_embeddings.py`:
```python
"""Freeze public-corpus embeddings for the no-model CI gate (PR3).

Embeds every public-corpus passage (``embed`` → passage role) and every public
golden question (``embed_query`` → query role) with the active embedder, then
writes TWO files — a numeric ``.npz`` plus a JSON string sidecar. The split
keeps the numpy loader on its safe default (no object arrays): strings inside
the archive would force the unsafe loader flag, an arbitrary-code-execution
risk on a committed fixture (spec 2026-06-06 §8).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.eval.adapter import _build_key_map
from app.eval.dataset import load_golden

LOGGER = logging.getLogger(__name__)

OUT_DIR = Path("data/eval/corpus_public")
GOLDEN = Path("data/eval/golden_public.jsonl")


@dataclass(frozen=True)
class FrozenSet:
    passage_keys: tuple[str, ...]
    passage_vecs: "np.ndarray"  # (N, d) float32, L2-normalized
    query_texts: list[str]
    query_vecs: "np.ndarray"  # (M, d) float32, L2-normalized


def _l2(rows: list[list[float]]) -> "np.ndarray":
    arr = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def build_frozen(store, embedder, golden_path: Path) -> FrozenSet:
    key_map = _build_key_map(store)
    with store._connect() as conn:  # noqa: SLF001 — same access the adapter uses
        rows = conn.execute(
            "SELECT document_id, chunk_index, text FROM kb_chunks "
            "ORDER BY document_id, chunk_index"
        ).fetchall()
    keys: list[str] = []
    passage_rows: list[list[float]] = []
    for doc_id, idx, text in rows:
        key = key_map.get((int(doc_id), int(idx)))
        if key is None:
            continue
        keys.append(key)
        passage_rows.append(embedder.embed(text))
        if len(keys) % 50 == 0:
            LOGGER.info("embedded %d passages", len(keys))

    questions = [it.question for it in load_golden(golden_path)]
    query_rows = [embedder.embed_query(q) for q in questions]
    return FrozenSet(tuple(keys), _l2(passage_rows), questions, _l2(query_rows))


def write_frozen(frozen: FrozenSet, out_dir: Path, embedder_tag: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"frozen_{embedder_tag}.npz"
    keys = out_dir / f"frozen_{embedder_tag}.keys.json"
    np.savez_compressed(npz, passage_vecs=frozen.passage_vecs, query_vecs=frozen.query_vecs)
    keys.write_text(
        json.dumps(
            {"passage_keys": list(frozen.passage_keys), "query_texts": frozen.query_texts},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return npz, keys


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=str(GOLDEN))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="bge-m3", help="embedder tag in the output filenames")
    args = parser.parse_args(argv)

    from app.services.kb_embeddings import get_embedder
    from app.services.kb_store import get_store

    frozen = build_frozen(get_store(), get_embedder(), Path(args.golden))
    npz, keys = write_frozen(frozen, Path(args.out_dir), args.tag)
    print(
        f"OK: {len(frozen.passage_keys)} passages, "
        f"{len(frozen.query_texts)} queries -> {npz}, {keys}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.13 -m pytest tests/scripts/test_build_frozen_embeddings.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint + commit the script**

```powershell
py -3.13 -m ruff check scripts/build_frozen_embeddings.py tests/scripts/test_build_frozen_embeddings.py
py -3.13 -m black scripts/build_frozen_embeddings.py tests/scripts/test_build_frozen_embeddings.py
git add scripts/build_frozen_embeddings.py tests/scripts/test_build_frozen_embeddings.py
git commit -m "feat(eval): frozen-embeddings builder (numeric npz + JSON keys sidecar)"
```

- [ ] **Step 6: Run the real freeze (operator step)**

```powershell
$env:KB_MVP_DB_PATH="var/data/kb_public.sqlite"
$env:KB_EMBEDDINGS_BACKEND="st"; $env:ST_EMBED_MODEL="BAAI/bge-m3"
py -3.13 -m scripts.build_frozen_embeddings
```
Expected: `OK: <N≥250> passages, <M≥55> queries -> data/eval/corpus_public/frozen_bge-m3.npz, ...keys.json`; `.npz` is roughly 1–3 MB.

- [ ] **Step 7: Sanity-check the frozen artifact ranks its own golden**

Run a local throwaway script (same env) that, for every answerable golden item, computes `sims = passage_vecs @ query_vecs[qi]`, takes the top-5 `passage_keys` by similarity, and counts items where any `relevant_chunks` key appears:
```python
import json
import numpy as np
from pathlib import Path
from app.eval.dataset import load_golden

v = np.load("data/eval/corpus_public/frozen_bge-m3.npz")
meta = json.loads(Path("data/eval/corpus_public/frozen_bge-m3.keys.json").read_text(encoding="utf-8"))
items = [it for it in load_golden(Path("data/eval/golden_public.jsonl")) if it.relevant_chunks]
qi = {t: i for i, t in enumerate(meta["query_texts"])}
hits = 0
for it in items:
    sims = v["passage_vecs"] @ v["query_vecs"][qi[it.question]]
    top5 = [meta["passage_keys"][i] for i in np.argsort(-sims)[:5]]
    hits += any(k in top5 for k in it.relevant_chunks)
print(f"hit@5 = {hits/len(items):.3f} over {len(items)} answerable items")
```
Expected: `hit@5` ≥ 0.6 (this number seeds PR3's threshold file — record it for the PR body). If it is near zero, the freeze and the store disagree — re-run Task 3 Step 1 and Task 6 Step 6 in the same env.

- [ ] **Step 8: Commit the frozen artifacts**

```powershell
git add data/eval/corpus_public/frozen_bge-m3.npz data/eval/corpus_public/frozen_bge-m3.keys.json
git commit -m "chore(eval): frozen bge-m3 vectors for the public corpus (CI gate input)"
```

---

## Task 7: Retire the contract golden to the private half

**Files:**
- Delete (git): `scripts/build_curated_golden.py`, `data/eval/golden_curated.jsonl`, `data/eval/golden_curated.sig.json`
- Modify: `scripts/eval_rag.py:147`, `tests/conftest.py:50,58-62`
- Rename/Modify: `tests/test_golden_curated.py` → `tests/test_golden_public.py`

- [ ] **Step 1: Preserve the private copies (gitignored location)**

```powershell
New-Item -ItemType Directory -Force var/data/eval/private | Out-Null
Copy-Item data/eval/golden_curated.jsonl, data/eval/golden_curated.sig.json var/data/eval/private/
Copy-Item scripts/build_curated_golden.py var/data/eval/private/build_private_golden.py
git check-ignore var/data/eval/private/golden_curated.jsonl   # must print the path
```
Expected: `git check-ignore` prints the path (confirming it stays out of git). If it prints nothing, STOP and add the ignore rule first.

- [ ] **Step 2: Rewrite the golden test for the public set**

`git mv tests/test_golden_curated.py tests/test_golden_public.py`, then replace its content:
```python
"""Validate the committed public golden set (structure, not retrieval quality)."""

from __future__ import annotations

from pathlib import Path

from app.eval.dataset import load_golden, read_signature

GOLDEN = Path("data/eval/golden_public.jsonl")


def test_public_golden_loads_and_uses_composite_keys() -> None:
    items = load_golden(GOLDEN)
    assert len(items) >= 50
    for item in items:
        for key in item.relevant_chunks:
            assert ":" in key, f"non-composite key {key!r} in {item.question!r}"


def test_public_golden_has_refusals_and_curated() -> None:
    items = load_golden(GOLDEN)
    assert sum(1 for it in items if it.expect_refusal) >= 3
    assert sum(1 for it in items if it.source == "curated") >= 15


def test_public_golden_signature_pins_st_1024() -> None:
    sig = read_signature(GOLDEN)
    assert sig is not None
    data = sig.to_dict()
    assert data["dim"] == 1024
    assert "st" in str(data["embedder_name"])
```
(Adjust the `to_dict` field names to the real `CorpusSignature` keys — confirm with `py -3.13 -c "from app.eval.dataset import read_signature; from pathlib import Path; print(read_signature(Path('data/eval/golden_public.jsonl')).to_dict())"`. Adjust the `>= 50` floor to the real item count if Task 4 produced fewer after review.)

- [ ] **Step 3: Repoint the default golden + the fixture guard**

In `scripts/eval_rag.py` line 147: `default="data/eval/golden_curated.jsonl"` → `default="data/eval/golden_public.jsonl"`.
In `tests/conftest.py` `_PROTECTED_FIXTURES`: replace the two `golden_curated` entries with:
```python
    REPO_ROOT / "data" / "eval" / "golden_public.jsonl",
    REPO_ROOT / "data" / "eval" / "golden_public.sig.json",
```
and update the comment block above it (drop the `build_curated_golden.py` reference, mention the `eval_rag.py run --golden` default = `golden_public`).

- [ ] **Step 4: Remove the retired files from git**

```powershell
git rm scripts/build_curated_golden.py data/eval/golden_curated.jsonl data/eval/golden_curated.sig.json
```

- [ ] **Step 5: Grep for stragglers**

Run: `git grep -n "golden_curated" -- ':!docs/superpowers'`
Expected: no hits in `app/`, `scripts/`, or `tests/` (historical references inside `docs/superpowers/` stay). Append a one-line note to `docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md`: «2026-06-10: golden_curated retired to the private half (`var/data/eval/private/`); the committed guard is now `data/eval/golden_public.jsonl`». Fix any hit outside docs.

- [ ] **Step 6: Run the affected tests**

Run: `py -3.13 -m pytest tests/test_golden_public.py tests/test_eval_cli.py -q; $LASTEXITCODE`
Expected: exit `0`.

- [ ] **Step 7: Commit**

```powershell
git add -A
git commit -m "refactor(eval)!: retire contract golden to private; public golden is the committed guard"
```

---

## Task 8: PR1 carry-over tidy — string keys in `test_eval_generation.py`

**Files:**
- Modify: `tests/test_eval_generation.py:48,59,62`

- [ ] **Step 1: Replace the int literals**

At the flagged lines, change `EvalHit(1, ...)` → `EvalHit("f.md:1", ...)` and `GoldenItem(..., (7,))` → `GoldenItem(..., ("7",))` (match each call's actual argument shape — these constructors now take `chunk_key: str` / `relevant_chunks: tuple[str, ...]`).

- [ ] **Step 2: Run the file**

Run: `py -3.13 -m pytest tests/test_eval_generation.py -q; $LASTEXITCODE`
Expected: exit `0`.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_eval_generation.py
git commit -m "test(eval): finish composite-key migration in generation tests"
```

---

## Final verification (whole-feature)

- [ ] **Step 1: Full suite, CI-mirrored**

Run: `py -3.13 -m pytest -q --ignore=backend; $LASTEXITCODE`
Expected: exit `0`.

- [ ] **Step 2: Lint/format/type the touched code**

```powershell
py -3.13 -m ruff check scripts/ tests/ app/
py -3.13 -m black --check scripts/build_public_corpus.py scripts/build_frozen_embeddings.py tests/scripts/ tests/test_golden_public.py
py -3.13 -m mypy app
```
Expected: ruff/black clean; mypy shows **no new** errors versus the 244-error baseline.

- [ ] **Step 3: PR**

Open a PR titled `feat(eval): synthetic public corpus + frozen embeddings (hybrid-eval PR2)`, body covering: corpus composition table, golden composition (auto/curated/refusals counts), the Task 6 Step 7 `hit@5` sanity number (seed for PR3 thresholds), and the §13 retirement note. Flag the review gate: synthetic docs + golden need a human pass for triviality/answer-leak before merge.

---

## Self-review notes (author)

- **Spec §6 coverage:** generator script (Task 1), 9 docs across the dog-food types (Task 2), determinism = committed text (Tasks 2–3), review gate (Task 2 Step 2, Task 4 Step 2, PR body), golden auto+curated with ≥3 refusals + sidecar (Tasks 4–5). ✓
- **Spec §8 inputs for PR3:** safe-loader `.npz` + keys sidecar + a measured `hit@5` seed number (Task 6). ✓
- **Spec §13 (accepted: retire):** Task 7, including the conftest guard swap and the `eval_rag` default repoint found by dependency grep. ✓
- **PR1 deferred items:** curated-golden int tuples die with the retired script (Task 7); generation-test ints fixed (Task 8). ✓
- **Type consistency:** `ingest_corpus(store, corpus_dir) -> int`, `build_frozen(store, embedder, golden_path) -> FrozenSet`, `write_frozen(frozen, out_dir, tag) -> (Path, Path)` used consistently across tasks; `FrozenSet` field names match PR3's planned `make_frozen_retriever` reader (`passage_vecs`, `query_vecs`, `passage_keys`, `query_texts`). ✓
- **Test-env gotcha:** default embedder is ST since #585 — every unit test here injects `embedder=` into `KnowledgeBaseStore`, so the cached `get_embedder()` is never touched.
