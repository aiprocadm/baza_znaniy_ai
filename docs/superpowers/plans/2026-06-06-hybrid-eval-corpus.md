# Hybrid Eval Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RAG answer-quality eval reproducible and trustworthy by giving chunks a stable identity, then committing a synthetic public corpus that gates CI deterministically while real documents stay private/local — and run the queued gates B/D + a judge baseline on the result.

**Architecture:** Two halves of one harness. The *public* half is synthetic RU documents + golden + frozen embeddings, committed so CI re-runs deterministic retrieval metrics with no model download. The *private* half is the operator's real documents, kept local for trustworthy absolute numbers and LLM-judge scoring. The keystone is a stable chunk identity `"<filename>:<chunk_index>"` that survives re-ingestion, replacing the fragile global autoincrement `kb_chunks.id`.

**Tech Stack:** Python 3.13 (`py -3` launcher, no venv), pytest (TDD), numpy (frozen-vector cosine), the in-process `st` embedder (BAAI/bge-m3) + GGUF eval provider (Qwen2.5-3B) from the local-keyless stack, GitHub Actions (path-scoped CI).

**Source spec:** [`docs/superpowers/specs/2026-06-06-hybrid-eval-corpus-design.md`](../specs/2026-06-06-hybrid-eval-corpus-design.md).

**Conventions (this repo):** run tests with `py -3 -m pytest ... --ignore=backend`; piping pytest drops the final summary, so confirm pass/fail by the **exit code** (`$LASTEXITCODE` in PowerShell). Commit messages follow Conventional Commits; PRs ≤400 LoC.

---

## Why PR1 is fully detailed and PR2–PR4 are a roadmap

PR1 (stable chunk identity) touches only existing code with known signatures, so it is written below as complete bite-sized TDD tasks. PR2–PR4 produce or consume **artifacts that do not exist until their predecessor runs**:

- PR2's golden keys (`"contract_services.md:3"`) only exist once the synthetic corpus is authored and ingested.
- PR3's committed CI threshold is *the metric numbers PR2's frozen vectors produce* — fabricating them now would be a placeholder.
- PR4's gate B/D deltas are *measurements*, not code.

Writing fake values for those would violate the no-placeholder rule. So PR2–PR4 are specified to roadmap fidelity (exact files, interfaces, the codeable pieces in full, and acceptance criteria), and **each is expanded into full bite-sized tasks when its predecessor merges** — at which point the real artifacts exist to write against. Execute PR1 first; then return to expand PR2.

---

## File Structure

**PR1 — modify (stable identity):**
- `app/eval/dataset.py` — `GoldenItem.relevant_chunks: tuple[str, ...]` (was `relevant_chunk_ids: tuple[int, ...]`); JSONL read/write of composite keys + legacy back-compat read.
- `app/eval/adapter.py` — `EvalHit.chunk_key: str` (was `chunk_id: int`); `_build_key_map` (join `kb_documents.filename`); new `build_global_id_key_map`; `make_retriever`/`make_mvp_*` rewired.
- `app/eval/metrics.py` — type hints `int → str` (logic unchanged; set-membership is type-agnostic).
- `app/eval/retrieval_eval.py` — read `item.relevant_chunks` / `h.chunk_key`.
- `scripts/eval_rag.py` — `cmd_generate` builds composite keys via `build_global_id_key_map`.
- `scripts/build_curated_golden.py` — convert its int chunk-ids to composite keys at build time.
- Tests: `tests/test_eval_dataset.py`, `tests/test_eval_adapter.py`, `tests/test_eval_retrieval.py`, `tests/test_eval_cli.py`, `tests/test_golden_curated.py`.

**PR2 — create (public corpus):** `scripts/build_public_corpus.py`, `scripts/build_frozen_embeddings.py`, `data/eval/corpus_public/*.md`, `data/eval/golden_public.jsonl` (+ `.sig.json`), `data/eval/corpus_public/frozen_bge-m3.npz` (numeric vectors only) + `data/eval/corpus_public/frozen_bge-m3.keys.json` (string keys/texts); modify `scripts/build_curated_golden.py` (retire contract → private).

**PR3 — create (CI gate):** `app/eval/frozen.py` (`make_frozen_retriever`), `data/eval/ci_thresholds.json`, `tests/test_eval_frozen.py`, a new job in `.github/workflows/ci.yml`.

**PR4 — modify (private half + gates):** `scripts/eval_rag.py` (or a small `app/eval/corpus_select.py`) for `KB_EVAL_CORPUS`; `tests/conftest.py` (extend `_protect_committed_fixtures`); `docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md` (append offline path + private-corpus instructions).

---

# PR1 — Stable chunk identity (executable now)

**Outcome:** the eval identifies chunks by `"<filename>:<chunk_index>"` everywhere; metrics are unchanged by construction; unit tests cover the new scheme with fake data (no corpus/model needed). Branch off the design branch:

```
git checkout -b feat/eval-stable-chunk-id
```

---

### Task 1: `GoldenItem` composite-key schema

**Files:**
- Modify: `app/eval/dataset.py`
- Test: `tests/test_eval_dataset.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_eval_dataset.py`:

```python
def test_golden_item_round_trips_composite_keys():
    from app.eval.dataset import GoldenItem
    item = GoldenItem(
        question="Сколько стоит услуга?",
        relevant_chunks=("contract.md:3", "contract.md:4"),
        reference_answer="45000",
        source="curated",
    )
    line = item.to_jsonl_line()
    back = GoldenItem.from_jsonl_line(line)
    assert back == item
    assert back.relevant_chunks == ("contract.md:3", "contract.md:4")


def test_golden_item_reads_legacy_int_labels_as_strings():
    # Old QAPair / int-labelled lines must still load (stringified, won't match
    # composite hits, but must not crash).
    from app.eval.dataset import GoldenItem
    legacy = '{"instruction":"q","input":"","output":"a","meta":{"relevant_chunk_ids":[7,12]}}'
    item = GoldenItem.from_jsonl_line(legacy)
    assert item.relevant_chunks == ("7", "12")

    legacy_qapair = '{"instruction":"q","input":"","output":"a","meta":{"source_chunk_id":7}}'
    assert GoldenItem.from_jsonl_line(legacy_qapair).relevant_chunks == ("7",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_dataset.py -k "composite_keys or legacy_int" -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'relevant_chunks'`.

- [ ] **Step 3: Write minimal implementation** — in `app/eval/dataset.py`, replace the `GoldenItem` dataclass body:

```python
@dataclass(frozen=True, slots=True)
class GoldenItem:
    question: str
    relevant_chunks: tuple[str, ...]
    reference_answer: str = ""
    expect_refusal: bool = False
    source: str = "auto"  # "auto" | "curated"

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction": self.question,
            "input": "",
            "output": self.reference_answer,
            "meta": {
                "relevant_chunks": [str(c) for c in self.relevant_chunks],
                "expect_refusal": bool(self.expect_refusal),
                "source": self.source,
            },
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False) + "\n"

    @classmethod
    def from_jsonl_line(cls, line: str) -> "GoldenItem":
        data = json.loads(line)
        meta = data.get("meta") or {}
        if "relevant_chunks" in meta:
            keys = [str(c) for c in meta["relevant_chunks"]]
        elif "relevant_chunk_ids" in meta:  # legacy int labels
            keys = [str(int(c)) for c in meta["relevant_chunk_ids"]]
        else:
            sid = meta.get("source_chunk_id")
            keys = [str(int(sid))] if sid is not None else []
        return cls(
            question=str(data["instruction"]),
            relevant_chunks=tuple(keys),
            reference_answer=str(data.get("output", "")),
            expect_refusal=bool(meta.get("expect_refusal", False)),
            source=str(meta.get("source", "auto")),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_dataset.py -k "composite_keys or legacy_int" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Update any existing assertions in this file** — search `tests/test_eval_dataset.py` for `relevant_chunk_ids=` and `relevant_chunk_ids ==`; replace constructor kwargs with `relevant_chunks=` using **string** values (e.g. `(7,)` → `("7",)`), and assertion reads with `.relevant_chunks`. Run the whole file:

Run: `py -3 -m pytest tests/test_eval_dataset.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```
git add app/eval/dataset.py tests/test_eval_dataset.py
git commit -m "refactor(eval): GoldenItem uses composite '<file>:<idx>' chunk keys (+legacy read)"
```

---

### Task 2: `EvalHit.chunk_key` + filename-aware id maps

**Files:**
- Modify: `app/eval/adapter.py`
- Test: `tests/test_eval_adapter.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_eval_adapter.py`:

```python
def test_build_key_map_joins_filename(tmp_path):
    # A throwaway store with one document and two chunks.
    from app.services.kb_store import KnowledgeBaseStore
    from app.eval.adapter import _build_key_map, build_global_id_key_map
    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"))
    doc_id = store.add_document(title="Contract", text="x", filename="contract.md")
    # add_document chunked "x" into one chunk; force a second for the test.
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO kb_chunks(document_id, chunk_index, text, embedding, embedder, dim) "
            "VALUES (?,?,?,?,?,?)",
            (doc_id, 1, "y", b"\x00" * 8, "hash", 2),
        )
        conn.commit()
    key_map = _build_key_map(store)
    assert key_map[(doc_id, 0)] == "contract.md:0"
    assert key_map[(doc_id, 1)] == "contract.md:1"
    gid = build_global_id_key_map(store)
    assert set(gid.values()) == {"contract.md:0", "contract.md:1"}


def test_make_retriever_emits_chunk_keys():
    from app.eval.adapter import make_retriever, EvalHit

    class _Hit:
        def __init__(self, doc, idx, text):
            self.document_id, self.chunk_index, self.text = doc, idx, text

    hits = [_Hit(1, 0, "a"), _Hit(1, 2, "b")]
    key_map = {(1, 0): "f.md:0", (1, 2): "f.md:2"}
    out = make_retriever(lambda q, k: hits[:k], key_map)("q", 5)
    assert [h.chunk_key for h in out] == ["f.md:0", "f.md:2"]
    assert isinstance(out[0], EvalHit)
```

> If `KnowledgeBaseStore.add_document` has a different name/signature, adjust the
> first test to the store's real ingest method (confirm in `app/services/kb_store.py`)
> — the second test (`make_retriever`) is pure and authoritative for the rename.

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_adapter.py -k "key_map or chunk_keys" -v`
Expected: FAIL — `ImportError: cannot import name '_build_key_map'` / `EvalHit` has no `chunk_key`.

- [ ] **Step 3: Write minimal implementation** — in `app/eval/adapter.py`, replace `EvalHit`, the map builders, `make_retriever`, and the `make_mvp_*` wrappers:

```python
@dataclass(frozen=True, slots=True)
class EvalHit:
    chunk_key: str  # composite "<filename>:<chunk_index>"
    text: str
    title: str = ""


Retriever = Callable[[str, int], Sequence[EvalHit]]


def _chunk_key(filename, document_id: int, chunk_index: int) -> str:
    base = filename or f"doc{document_id}"
    return f"{base}:{chunk_index}"


def make_retriever(
    search: Callable[[str, int], Sequence["_SearchHitLike"]],
    key_map: Mapping[tuple[int, int], str],
) -> Retriever:
    def _retrieve(query: str, top_k: int) -> list[EvalHit]:
        out: list[EvalHit] = []
        for h in search(query, top_k):
            key = key_map.get((int(h.document_id), int(h.chunk_index)))
            if key is None:
                continue
            title = getattr(h, "filename", "") or getattr(h, "document_title", "") or ""
            out.append(EvalHit(chunk_key=key, text=h.text, title=title))
        return out

    return _retrieve


def _build_key_map(store) -> dict[tuple[int, int], str]:
    with store._connect() as conn:  # noqa: SLF001
        rows = conn.execute(
            "SELECT c.document_id, c.chunk_index, d.filename "
            "FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id"
        ).fetchall()
    return {
        (int(doc_id), int(idx)): _chunk_key(fn, int(doc_id), int(idx))
        for doc_id, idx, fn in rows
    }


def build_global_id_key_map(store) -> dict[int, str]:
    """global kb_chunks.id -> composite key. For converting int-labelled goldens."""
    with store._connect() as conn:  # noqa: SLF001
        rows = conn.execute(
            "SELECT c.id, c.document_id, c.chunk_index, d.filename "
            "FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id"
        ).fetchall()
    return {
        int(cid): _chunk_key(fn, int(doc_id), int(idx))
        for cid, doc_id, idx, fn in rows
    }


def make_mvp_retriever(store) -> Retriever:
    """Wrap a live ``KnowledgeBaseStore`` as an eval Retriever."""
    return make_retriever(lambda q, k: store.search(q, top_k=k), _build_key_map(store))
```

Then in `make_mvp_reranking_retriever`, change the final line `_build_id_map(store)` → `_build_key_map(store)`. Delete the old `_build_id_map`. (`compute_signature` is unchanged — `max_chunk_id` remains a valid drift fingerprint.)

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_adapter.py -k "key_map or chunk_keys" -v`
Expected: PASS.

- [ ] **Step 5: Update existing assertions in this file** — search `tests/test_eval_adapter.py` for `_build_id_map`, `chunk_id=`, `.chunk_id` and replace with `_build_key_map`, `chunk_key=` (string values), `.chunk_key`. Run:

Run: `py -3 -m pytest tests/test_eval_adapter.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```
git add app/eval/adapter.py tests/test_eval_adapter.py
git commit -m "refactor(eval): EvalHit.chunk_key + filename-joined id maps"
```

---

### Task 3: Rewire `retrieval_eval` + `metrics` types

**Files:**
- Modify: `app/eval/retrieval_eval.py`, `app/eval/metrics.py`
- Test: `tests/test_eval_retrieval.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_eval_retrieval.py`:

```python
def test_evaluate_item_scores_on_chunk_keys():
    from app.eval.dataset import GoldenItem
    from app.eval.adapter import EvalHit
    from app.eval.retrieval_eval import evaluate_item

    item = GoldenItem("q", relevant_chunks=("f.md:2",), reference_answer="")
    retriever = lambda q, k: [EvalHit("f.md:0", "a"), EvalHit("f.md:2", "b")][:k]
    scores = evaluate_item(item, retriever, max_k=5)
    assert scores["hit@5"] == 1.0
    assert scores["hit@1"] == 0.0  # relevant key is at rank 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_retrieval.py -k chunk_keys -v`
Expected: FAIL — `AttributeError: 'GoldenItem' object has no attribute 'relevant_chunk_ids'` (current `evaluate_item` reads the old name).

- [ ] **Step 3: Write minimal implementation** — in `app/eval/retrieval_eval.py`, change `evaluate_item`:

```python
def evaluate_item(item: GoldenItem, retriever: Retriever, *, max_k: int) -> dict[str, float]:
    hits = retriever(item.question, max_k)
    retrieved = [h.chunk_key for h in hits]
    return score_item(item.relevant_chunks, retrieved)
```

In `app/eval/metrics.py`, change the four signatures' `Collection[int]`/`Sequence[int]` to `Collection[str]`/`Sequence[str]` (no logic change). Update the module docstring line to "Pure retrieval-quality metrics over chunk keys."

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_retrieval.py tests/test_eval_metrics.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```
git add app/eval/retrieval_eval.py app/eval/metrics.py tests/test_eval_retrieval.py
git commit -m "refactor(eval): score retrieval on chunk keys"
```

---

### Task 4: `eval_rag generate` builds composite keys

**Files:**
- Modify: `scripts/eval_rag.py`
- Test: `tests/test_eval_cli.py`

- [ ] **Step 1: Write the failing test** — add to `tests/test_eval_cli.py` (follow the file's existing fake-store/fake-provider fixtures; the assertion is the new behavior):

```python
def test_generate_emits_composite_keys(monkeypatch, tmp_path):
    # Build a tiny real store so build_global_id_key_map has a filename to join.
    from app.services.kb_store import KnowledgeBaseStore
    import app.services.kb_store as kb_store_mod
    import scripts.eval_rag as cli
    store = KnowledgeBaseStore(str(tmp_path / "kb.sqlite"))
    store.add_document(title="Doc", text="первый абзац про оплату услуг.", filename="doc.md")
    monkeypatch.setattr(kb_store_mod, "get_store", lambda: store)
    monkeypatch.setattr(cli, "get_store", lambda: store)

    class _Pair:
        instruction = "Сколько стоит?"
        output = "45000"
        source_chunk_id = 1  # global id of the first chunk

    class _Gen:
        name, model = "fake", "fake"
        def generate_for_chunk(self, **kw): return [_Pair()]
    monkeypatch.setattr(cli, "_gen_provider", lambda: _Gen())
    monkeypatch.setattr(cli.sq, "estimate_total_cost_usd", lambda **kw: 0.0)

    out = tmp_path / "golden_auto.jsonl"
    cli.main(["generate", "--out", str(out), "--limit", "1", "--yes"])
    from app.eval.dataset import load_golden
    items = load_golden(out)
    assert items and all(":" in k for it in items for k in it.relevant_chunks)
```

> Mirror whatever fake-provider pattern `tests/test_eval_cli.py` already uses; the
> single load-bearing assertion is that emitted `relevant_chunks` are composite keys.

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3 -m pytest tests/test_eval_cli.py -k composite_keys -v`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'relevant_chunk_ids'` (current `cmd_generate` still builds the old field).

- [ ] **Step 3: Write minimal implementation** — in `scripts/eval_rag.py`: add `build_global_id_key_map` to the adapter import line, then in `cmd_generate` build the map once and use it:

```python
    generator = sq.SyntheticQAGenerator(provider=provider)
    key_map = build_global_id_key_map(store)
    items: list[GoldenItem] = []
    for chunk_id, text in chunks:
        for pair in generator.generate_for_chunk(
            chunks=[text], chunk_ids=[chunk_id], mode=sq.GenerationMode.SINGLE
        ):
            key = key_map.get(pair.source_chunk_id)
            if key is None:
                continue
            items.append(
                GoldenItem(
                    question=pair.instruction,
                    relevant_chunks=(key,),
                    reference_answer=pair.output,
                    expect_refusal=False,
                    source="auto",
                )
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3 -m pytest tests/test_eval_cli.py -v`
Expected: PASS (all in file).

- [ ] **Step 5: Commit**

```
git add scripts/eval_rag.py tests/test_eval_cli.py
git commit -m "refactor(eval): generate emits composite chunk keys"
```

---

### Task 5: Migrate `build_curated_golden.py` (ops script)

**Files:**
- Modify: `scripts/build_curated_golden.py`
- Test: `tests/test_golden_curated.py`

- [ ] **Step 1: Restructure the item lists to (question, int-ids, answer, expect_refusal)** — change `ANSWERABLE`/`REFUSALS` from `GoldenItem(...)` literals to plain tuples so ints can be converted at build time. Replace each `GoldenItem("Q", (3,), "A", source="curated")` with `("Q", (3,), "A", False)`; each refusal `GoldenItem("Q", (), "", expect_refusal=True, source="curated")` with `("Q", (), "", True)`. Then replace the imports + `main`:

```python
from app.eval.adapter import build_global_id_key_map, compute_signature
from app.eval.dataset import GoldenItem, save_golden, write_signature
from app.services.kb_store import get_store

def main() -> None:
    store = get_store()
    key_map = build_global_id_key_map(store)
    items = []
    for question, int_ids, answer, refuse in ANSWERABLE + REFUSALS:
        keys = tuple(key_map[i] for i in int_ids)  # KeyError = corpus drift, fail loud
        items.append(
            GoldenItem(question, keys, answer, expect_refusal=refuse, source="curated")
        )
    save_golden(GOLDEN, items)
    write_signature(GOLDEN, compute_signature(store))
    print(f"Wrote {len(items)} curated items + signature to {GOLDEN}")
```

- [ ] **Step 2: Update `tests/test_golden_curated.py`** — this test validates the committed golden. Change any assertion that expects integer `relevant_chunk_ids` to expect composite-key strings (`":" in key`). Run:

Run: `py -3 -m pytest tests/test_golden_curated.py -v`
Expected: PASS if the test only checks structure; if it asserts composite keys it may FAIL here until Step 3 regenerates the committed file — that is expected.

- [ ] **Step 3: Operator regenerates the committed golden** (requires the contract in the local store):

Run: `py -3 -m scripts.build_curated_golden`
Expected output: `Wrote 21 curated items + signature to data/eval/golden_curated.jsonl`. Then `git diff data/eval/golden_curated.jsonl` shows `relevant_chunk_ids` integers replaced by `relevant_chunks` composite keys.

Run: `py -3 -m pytest tests/test_golden_curated.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```
git add scripts/build_curated_golden.py tests/test_golden_curated.py data/eval/golden_curated.jsonl data/eval/golden_curated.sig.json
git commit -m "refactor(eval): curated golden uses composite chunk keys"
```

---

### Task 6: Full-suite green + grep for stragglers

- [ ] **Step 1: Grep for any remaining old producers** — search under `app/eval/`, `scripts/eval_rag.py`, `tests/test_eval_*` for `relevant_chunk_ids=`, `\.chunk_id\b`, `_build_id_map`. There should be **no remaining producers**; the legacy *reader* support in `dataset.from_jsonl_line` is intentional and stays.

- [ ] **Step 2: Run the eval suite + a broad slice**

Run: `py -3 -m pytest tests/ -k "eval or golden" --ignore=backend; $LASTEXITCODE`
Expected: exit code `0`. (Piping hides the summary line — trust the exit code.)

- [ ] **Step 3: Commit any straggler fixes**

```
git add -A
git commit -m "test(eval): finish composite-key migration across eval tests"
```

**PR1 done.** Open a PR titled `refactor(eval): stable composite chunk identity for the eval harness`. Net behavior unchanged; the foundation for PR2–PR4 is in place.

---

# PR2 — Synthetic public corpus (roadmap; expand after PR1 merges)

**Goal:** A committed, reproducible public corpus + golden, and the frozen embeddings PR3 will gate on. Retire the contract Q&A to the private half.

**Files & responsibilities:**
- `scripts/build_public_corpus.py` — author ~8–15 RU docs via the configured LLM (`select_provider`, default the local GGUF), write them as `data/eval/corpus_public/*.md` with stable filenames (`contract_services.md`, `nda.md`, `reglament_ot.md`, `procedure_onboarding.md`, `npa_excerpt.md`, …). One-time authoring step; the **committed artifact is the `.md` text**, not the generator run.
- `scripts/build_frozen_embeddings.py` — load the `st` embedder (bge-m3) via `app.services.kb_embeddings.get_embedder()`; embed every public-corpus passage and every `golden_public` question. Write **two files, no pickle**: `frozen_bge-m3.npz` holding only float32 arrays `passage_vecs[N,d]`, `query_vecs[M,d]` (L2-normalized), and `frozen_bge-m3.keys.json` holding `{"passage_keys": [str]*N, "query_texts": [str]*M}`. Splitting strings out of the `.npz` keeps `np.load` pickle-free (object arrays would require `allow_pickle=True` — an arbitrary-code-execution risk on a committed fixture).
- `data/eval/golden_public.jsonl` (+ `.sig.json`) — auto breadth via `eval_rag generate` over the public store + curated hard cases (multi-hop, paraphrase, ≥3 `expect_refusal`), authored as composite keys (PR1 scheme).
- Modify `scripts/build_curated_golden.py` — remove the 21 contract items (retire to a local private golden per spec §13); repoint the script at the public curated set, or rename to `build_private_golden.py` kept local. **Decision (spec §13, accepted): retire to private.**

**Build sequence (each a commit):**
1. Write & review synthetic docs → commit `data/eval/corpus_public/*.md`. Acceptance: a human review confirms no triviality/answer-leak; docs span the dog-food types.
2. Ingest publicly: `py -3 -m scripts.kb_cli reindex --embedder st --force-yes` against a **public** store (`KB_MVP_DB_PATH=var/data/kb_public.sqlite`). Acceptance: `compute_signature` reports `embedder_name="st", dim=1024`, doc_count = number of docs.
3. `py -3 -m scripts.eval_rag generate --out data/eval/golden_public.jsonl --limit 200` then hand-add curated items. Acceptance: ≥3 `expect_refusal`; all keys composite; `.sig.json` matches the public store.
4. `py -3 -m scripts.build_frozen_embeddings`. Acceptance: `.npz` loads **without `allow_pickle`** and holds only float arrays; `.keys.json` `passage_keys` length = corpus chunk count; vectors L2-normalized.
5. Retire contract golden to private; commit.

**Deferred from PR1 (carry into this PR — flagged by the PR1 final review):**
- `scripts/build_curated_golden.py:22-139` still passes **int** tuples to `GoldenItem.relevant_chunks` (now `tuple[str, ...]`). Harmless today (`to_dict` coerces via `str(...)`, reproducing the int-format file; it lives under `scripts/` so the gated `mypy app` doesn't see it), but migrate to composite keys (or convert via `build_global_id_key_map`) when this script is retired/repointed here.
- `tests/test_eval_generation.py:48,59,62` still constructs `EvalHit(1, ...)` / `GoldenItem(..., (7,))` with ints — inert (generation_eval never matches on `chunk_key`/`relevant_chunks`), but tidy to string keys here.

**Acceptance for the PR:** a fresh clone has the public `.md` + golden + `.npz` + `.keys.json`; nothing private is committed; `build_*` scripts are added to fixture protection in PR4.

> Expand this section into bite-sized TDD tasks once PR1 is merged and the real
> `kb_embeddings.get_embedder()` / store-ingest signatures are confirmed in-tree.

---

# PR3 — Frozen-embeddings CI gate (roadmap; expand after PR2 merges)

**Goal:** Every relevant PR re-runs deterministic retrieval metrics on the public corpus in CI, with no model download.

**Codeable now — `app/eval/frozen.py` (pickle-free):**

```python
"""A Retriever backed by precomputed (frozen) embeddings — no model load.

Loads committed numeric vectors (.npz) + a JSON sidecar of string keys/texts and
ranks passages by cosine similarity. Pure numpy, no pickle: deterministic, fast,
CI-safe. The .npz holds ONLY float arrays (no object dtype), so np.load needs no
allow_pickle — strings live in the JSON sidecar.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from app.eval.adapter import EvalHit, Retriever


def make_frozen_retriever(npz_path: str | Path, keys_path: str | Path) -> Retriever:
    vecs = np.load(Path(npz_path))  # numeric-only arrays; no allow_pickle
    pvecs = np.asarray(vecs["passage_vecs"], dtype=np.float32)  # L2-normalized
    qvecs = np.asarray(vecs["query_vecs"], dtype=np.float32)
    meta = json.loads(Path(keys_path).read_text(encoding="utf-8"))
    pkeys: list[str] = list(meta["passage_keys"])
    q_index = {t: i for i, t in enumerate(meta["query_texts"])}

    def _retrieve(query: str, top_k: int) -> list[EvalHit]:
        qi = q_index.get(query)
        if qi is None:
            raise KeyError(f"query not in frozen set: {query!r}")
        sims = pvecs @ qvecs[qi]
        order = np.argsort(-sims)[:top_k]
        return [EvalHit(chunk_key=pkeys[i], text="", title="") for i in order]

    return _retrieve
```

**Other files:**
- `data/eval/ci_thresholds.json` — committed floor per metric (`{"hit@1": <PR2 number − margin>, "mrr@5": ...}`). Values are *the numbers PR2's frozen run produces*, minus a small margin — written when PR2 exists.
- `tests/test_eval_frozen.py` — (a) pure test: a tiny hand-built `.npz` + `.keys.json` in `tmp_path`, assert ranking by cosine; (b) gate test: load the committed `.npz`/`.keys.json`, run `retrieval_eval.evaluate` over `golden_public` (answerable items only), assert each aggregate ≥ `ci_thresholds.json`; (c) `@pytest.mark.integration` staleness test: re-encode 2–3 passages with real bge-m3, assert cosine match to the committed vectors within 1e-3.
- `.github/workflows/ci.yml` — new job `eval-gate` (mirrors the `path-classifier` + app-job pattern), path-scoped to `app/eval/**`, `app/retriever/**`, `app/services/kb_store.py`, `data/eval/**`; runs `py -3 -m pytest tests/test_eval_frozen.py -m "not integration"`.

**Acceptance:** CI fails when a change drops a public retrieval metric below its committed floor; no model is downloaded in the gate job.

> Expand into bite-sized tasks once PR2 has produced `golden_public` + the `.npz`
> (the threshold values and the gate test's expected numbers come from that run).

---

# PR4 — Private half + run gates B/D + judge baseline (roadmap; expand after PR3 merges)

**Goal:** A clean public/private switch, fixture-write protection for the new committed files, and the actual measured gate results on the private corpus.

**Files:**
- `app/eval/corpus_select.py` (or inline in `scripts/eval_rag.py`) — read `KB_EVAL_CORPUS=public|private` (default `public`); resolve `(store_db_path, golden_path)`. Private paths come from env (`KB_EVAL_PRIVATE_DB`, `KB_EVAL_PRIVATE_GOLDEN`), never committed.
- `tests/conftest.py` — extend `_protect_committed_fixtures` to also snapshot `data/eval/corpus_public/` (`*.md`, `*.npz`, `*.keys.json`) and `data/eval/golden_public.jsonl` (+ sidecar) so a stray test write fails loudly.
- `tests/test_eval_private.py` — `@pytest.mark.integration`; **skips loudly** (with reason) when `KB_EVAL_PRIVATE_DB` is unset/absent.
- `docs/superpowers/runbooks/2026-06-05-eval-baseline-and-gates.md` — append the fully-offline path (`KB_EMBEDDINGS_BACKEND=st`, `KB_LLM_PROVIDER=gguf`) and private-corpus selection.

**Measured gate runs (operator, on the private corpus — produce, then record):**
1. Baseline: `KB_EVAL_CORPUS=private … py -3 -m scripts.eval_rag run --judge --out var/data/eval/baseline.json`. Keep `baseline.json`.
2. **Gate B (e5):** reindex private under e5 + `VECTOR_E5_PREFIX=true` → `run --out var/data/eval/gateB.json` → `compare baseline.json gateB.json`. Keep iff `recall@k`/`mrr@k` ↑; paste delta table into the PR.
3. **Gate D (top_k):** `py -3 -m scripts.eval_sweep --golden <private> --values 5,8,10,12 --judge`. Set MVP `ask` `top_k` to argmax `completeness` without dropping `faithfulness`; commit with the table.

**Acceptance:** private tests skip loudly without a corpus; new committed fixtures are write-protected; gate B/D decisions are each backed by a `compare` delta table recorded in the runbook (the gate-C discipline).

> Expand into bite-sized tasks once PR3's CI gate is green and the private corpus
> exists locally (gate deltas are measurements, recorded — not code).

---

## Plan self-review

- **Spec coverage:** §5 stable identity → PR1 (Tasks 1–6). §6 public synthetic corpus → PR2. §7 private half → PR4. §8 frozen-embeddings CI gate → PR3. §9 gates B/D + baseline → PR4. §10 guardrails (fixture protection, sig drift, corpus-absent) → PR4 + existing `run` drift guard (unchanged). §11 testing (DI, integration markers) → throughout. §13 retire-to-private → PR2 step 5. All sections mapped.
- **Placeholder scan:** PR1 contains complete code/tests/commands. PR2–PR4 deliberately defer *data-dependent values* (corpus content, threshold numbers, gate deltas) to expansion-after-predecessor, with acceptance criteria in their place — these are artifacts that cannot exist before the predecessor runs, not unfilled blanks.
- **Type consistency:** `relevant_chunks: tuple[str, ...]`, `EvalHit.chunk_key: str`, `_build_key_map`, `build_global_id_key_map`, `make_frozen_retriever(npz_path, keys_path)` are used consistently across PR1 tasks and the PR3 retriever. `compute_signature` left unchanged.
- **Security:** the frozen retriever is pickle-free (`.npz` numeric-only + JSON sidecar) — no `allow_pickle`, so a tampered committed fixture cannot execute code.
- **Known confirm-at-impl points (flagged inline, not placeholders):** `KnowledgeBaseStore.add_document` signature (Task 2/4 tests) and `kb_embeddings.get_embedder()` (PR2) — the load-bearing assertions don't depend on them; confirm against the real files when executing.
