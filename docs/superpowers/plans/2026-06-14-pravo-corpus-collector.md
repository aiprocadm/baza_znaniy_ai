# pravo.gov.ru Corpus Collector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command reproducibly assembles ~20 Russian Federation codes into a clean Russian-text corpus (`corpus.jsonl` + `corpus.txt`) with a provenance `manifest.json`, usable by the three downstream sub-projects (mini-GPT, LoRA, RAG ingest).

**Architecture:** A small, import-safe Python package under `experiments/pravo_nn/` (NOT the production `app/` tree). Data flows `config → fetch (cached raw) → extract (clean Articles) → assemble (corpus + manifest)`. The *source of text* is an intentionally swappable detail decided by a mandatory spike (Task 3); everything downstream of `extract` is source-independent and built first.

**Tech Stack:** Python 3.13 (`py -3.13`), stdlib only (`urllib`, `json`, `hashlib`, `re`, `dataclasses`); pytest with the network mocked. No heavy deps — this is a data tool, not an ML one.

**Reference spec:** [docs/superpowers/specs/2026-06-14-pravo-corpus-collector-design.md](../specs/2026-06-14-pravo-corpus-collector-design.md)

**Conventions (from CLAUDE.md / repo memory):**
- Run tests with `py -3.13 -m pytest experiments/pravo_nn/tests` (bare `py -3` resolves to a 3.14 with no pytest).
- Lives under `experiments/` so CI's `app/**` path-scoped gates are untouched and the anti-roadmap (own-LLM rejected as a *product*) is not violated — this is research.
- Conventional Commits; commit after every green step.

---

## File Structure

```
experiments/
  __init__.py                              # makes experiments importable
  pravo_nn/
    __init__.py
    corpus_collector/
      __init__.py
      config.py        # CodeSpec dataclass + CODES list (~20 codes) + SOURCE_BASE
      fetch.py         # network: url_for(), fetch_raw() with on-disk cache + retry/backoff
      extract.py       # Article dataclass + strip_to_text/normalize/split_articles/extract_articles
      assemble.py      # write_corpus(), build_manifest(), MIN_ARTICLE_CHARS canary
      cli.py           # `collect` subcommand wiring fetch→extract→assemble
      spike.py         # THROWAWAY exploration (Task 3); deleted or kept as a script, not imported
    tests/
      __init__.py
      conftest.py      # adds repo root to sys.path if needed
      fixtures/
        sample_raw.txt # ONE committed raw document in the spike's chosen format
      test_config.py
      test_extract.py
      test_assemble.py
      test_manifest.py
      test_idempotent.py
      test_fetch.py
    data/              # raw/ (gitignored), corpus/, manifest.json (gitignored)
    README.md
    .gitignore
```

Responsibilities: `config` knows *which* codes (stable list). `fetch` knows *where* text comes from (source-specific, post-spike). `extract` turns raw bytes into clean `Article`s (source-specific shim + source-independent splitter). `assemble` writes the output contract. `cli` orchestrates. They share exactly one contract: the `Article` dataclass + the `data/corpus/` files.

---

## Task 1: Package scaffold + gitignore + README skeleton

**Files:**
- Create: `experiments/__init__.py` (empty)
- Create: `experiments/pravo_nn/__init__.py` (empty)
- Create: `experiments/pravo_nn/corpus_collector/__init__.py` (empty)
- Create: `experiments/pravo_nn/tests/__init__.py` (empty)
- Create: `experiments/pravo_nn/tests/conftest.py`
- Create: `experiments/pravo_nn/.gitignore`
- Create: `experiments/pravo_nn/README.md`

- [ ] **Step 1: Create the empty package markers**

Create `experiments/__init__.py`, `experiments/pravo_nn/__init__.py`, `experiments/pravo_nn/corpus_collector/__init__.py`, `experiments/pravo_nn/tests/__init__.py` — all empty files.

- [ ] **Step 2: Create `experiments/pravo_nn/tests/conftest.py`**

```python
"""Ensure the repo root is importable so `experiments.pravo_nn.*` resolves
when pytest is invoked from anywhere."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

- [ ] **Step 3: Create `experiments/pravo_nn/.gitignore`**

```gitignore
# Raw network responses are a rebuildable cache, not source.
data/raw/
# The assembled corpus + manifest are generated artifacts (tens of MB).
data/corpus/
data/manifest.json
```

- [ ] **Step 4: Create `experiments/pravo_nn/README.md`**

```markdown
# pravo.gov.ru corpus collector (sub-project 0)

Assembles ~20 Russian Federation codes into a clean Russian-text corpus for the
mini-GPT / LoRA / RAG sub-projects. See the design spec:
`docs/superpowers/specs/2026-06-14-pravo-corpus-collector-design.md`.

## Run

```
py -3.13 -m experiments.pravo_nn.corpus_collector.cli collect
py -3.13 -m pytest experiments/pravo_nn/tests
```

## Text source

> **DECIDED IN THE SPIKE (Task 3) — fill this in:** source = ___, because ___.
> Semantics: ___ (point-in-time as-published vs current consolidated).
```

- [ ] **Step 5: Verify the package imports**

Run: `py -3.13 -c "import experiments.pravo_nn.corpus_collector"`
Expected: no output, exit 0.

- [ ] **Step 6: Commit**

```bash
git add experiments/__init__.py experiments/pravo_nn experiments/pravo_nn/tests experiments/pravo_nn/.gitignore experiments/pravo_nn/README.md
git commit -m "chore(pravo-nn): scaffold corpus_collector package"
```

---

## Task 2: `config.py` — the list of codes

**Files:**
- Create: `experiments/pravo_nn/corpus_collector/config.py`
- Test: `experiments/pravo_nn/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
from experiments.pravo_nn.corpus_collector.config import CODES, CodeSpec


def test_codes_are_codespecs_with_nonempty_fields():
    assert len(CODES) >= 18  # ~20 RF codes
    for spec in CODES:
        assert isinstance(spec, CodeSpec)
        assert spec.name.strip()
        assert spec.slug.strip()


def test_slugs_are_unique_and_filename_safe():
    slugs = [s.slug for s in CODES]
    assert len(slugs) == len(set(slugs))  # no duplicates
    for slug in slugs:
        assert all(c.isalnum() or c == "-" for c in slug), slug
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: ... config` / `cannot import name 'CODES'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""The fixed list of Russian Federation codes to collect.

`name` is the canonical short legal name (lands in the corpus). `slug` is a
filename-safe id used for the on-disk raw cache. The *source URL* is NOT here —
it depends on the source chosen by the Task 3 spike and is built in fetch.py.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeSpec:
    name: str  # e.g. "ГК РФ"
    slug: str  # e.g. "gk-rf"


CODES: tuple[CodeSpec, ...] = (
    CodeSpec("ГК РФ", "gk-rf"),       # Гражданский
    CodeSpec("УК РФ", "uk-rf"),       # Уголовный
    CodeSpec("НК РФ", "nk-rf"),       # Налоговый
    CodeSpec("ТК РФ", "tk-rf"),       # Трудовой
    CodeSpec("КоАП РФ", "koap-rf"),   # Об административных правонарушениях
    CodeSpec("ЖК РФ", "zhk-rf"),      # Жилищный
    CodeSpec("СК РФ", "sk-rf"),       # Семейный
    CodeSpec("ГПК РФ", "gpk-rf"),     # Гражданский процессуальный
    CodeSpec("УПК РФ", "upk-rf"),     # Уголовно-процессуальный
    CodeSpec("АПК РФ", "apk-rf"),     # Арбитражный процессуальный
    CodeSpec("БК РФ", "bk-rf"),       # Бюджетный
    CodeSpec("ЗК РФ", "zk-rf"),       # Земельный
    CodeSpec("УИК РФ", "uik-rf"),     # Уголовно-исполнительный
    CodeSpec("КАС РФ", "kas-rf"),     # Административного судопроизводства
    CodeSpec("ГрК РФ", "grk-rf"),     # Градостроительный
    CodeSpec("ВК РФ", "vk-rf"),       # Водный
    CodeSpec("ЛК РФ", "lk-rf"),       # Лесной
    CodeSpec("ВзК РФ", "vzk-rf"),     # Воздушный
    CodeSpec("КТМ РФ", "ktm-rf"),     # Торгового мореплавания
)

# The base of the chosen text source. Empty until the Task 3 spike decides it;
# Task 8 (fetch) sets it to the spike's committed value.
SOURCE_BASE: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_config.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/corpus_collector/config.py experiments/pravo_nn/tests/test_config.py
git commit -m "feat(pravo-nn): config with the list of RF codes to collect"
```

---

## Task 3: SPIKE — decide the text source (decision gate, NOT TDD)

This is a deliberate exception to TDD — it is throwaway exploration whose only deliverables are a *decision*, a committed *fixture*, and a *README note*. Per the spec, `fetch.py`/`extract.py` are built against the chosen source, so this gate runs before Tasks 4 and 8.

**Files:**
- Create (throwaway): `experiments/pravo_nn/corpus_collector/spike.py`
- Create (committed): `experiments/pravo_nn/tests/fixtures/sample_raw.txt`
- Modify: `experiments/pravo_nn/README.md` (fill the "Text source" section)

- [ ] **Step 1: Write the exploration script**

```python
"""THROWAWAY spike: fetch 2-3 codes from each candidate source and report which
gives the cleanest text. Not imported by anything; not unit-tested. Delete or
keep as a one-off after the decision is committed to the README.

Candidates (see spec §Step 0):
  A. API  publication.pravo.gov.ru          (JSON; RISK: PDF scans -> OCR)
  B. ИПС  pravo.gov.ru/ips/                  (HTML/text)
  C. mirror data.apicrafter.ru/.../pubpravogovru  (already-parsed structured text)
"""

from __future__ import annotations

import sys
import urllib.request


def report(label: str, url: str) -> None:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            ctype = resp.headers.get("Content-Type", "?")
            raw = resp.read()
    except Exception as exc:  # spike: any failure is just a data point
        print(f"[{label}] FAILED {url}: {exc}")
        return
    head = raw[:4]
    is_pdf = head == b"%PDF"
    text = raw.decode("utf-8", errors="replace")
    n = len(text)
    n_articles = text.count("Стать")  # crude "looks like a code" signal
    print(
        f"[{label}] {url}\n"
        f"  content-type={ctype} bytes={len(raw)} is_pdf={is_pdf} "
        f"chars={n} 'Стать'_hits={n_articles}"
    )


if __name__ == "__main__":
    # Fill these with real candidate URLs for ГК/УК before running.
    candidates = {
        "A-api": sys.argv[1] if len(sys.argv) > 1 else "",
        "B-ips": sys.argv[2] if len(sys.argv) > 2 else "",
        "C-mirror": sys.argv[3] if len(sys.argv) > 3 else "",
    }
    for label, url in candidates.items():
        if url:
            report(label, url)
```

- [ ] **Step 2: Run the spike against real URLs**

Run (example): `py -3.13 -m experiments.pravo_nn.corpus_collector.spike "<api-url>" "<ips-url>" "<mirror-url>"`
Inspect the output. Decision rule: prefer the source with `is_pdf=False`, high `chars`, and many `'Стать'_hits`. If only the API works and it returns `is_pdf=True`, record that OCR is required and STOP to renegotiate scope before continuing (this is the project's biggest risk per the spec).

- [ ] **Step 3: Commit ONE raw fixture in the chosen format**

Save a short real excerpt (one code, a handful of articles) from the chosen source to `experiments/pravo_nn/tests/fixtures/sample_raw.txt`. This is the ground truth `test_extract` (Task 4) asserts against. Keep it small (a few KB) but representative — include at least two `Статья N` markers and any page-number / header noise the source carries.

- [ ] **Step 4: Record the decision in the README**

Edit `experiments/pravo_nn/README.md` "Text source" section: `source = <A/B/C + URL pattern>, because <cleanest per spike>. Semantics: <as-published point-in-time | current consolidated>.` The semantics note matters for sub-projects 2/2b (facts).

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/corpus_collector/spike.py experiments/pravo_nn/tests/fixtures/sample_raw.txt experiments/pravo_nn/README.md
git commit -m "spike(pravo-nn): choose text source; commit raw fixture + decision note"
```

---

## Task 4: `extract.py` — raw document → clean Articles

The `Article` dataclass + `normalize_whitespace` + `split_articles` are **source-independent** and fully concrete. `strip_to_text` is the thin source-specific shim — the version below handles an HTML source (crude tag strip); if the spike chose plain text, its body becomes a passthrough, and if it chose JSON, it pulls the text field. The `extract_articles` contract (`raw str → list[Article]`) does not change.

**Files:**
- Create: `experiments/pravo_nn/corpus_collector/extract.py`
- Test: `experiments/pravo_nn/tests/test_extract.py`

- [ ] **Step 1: Write the failing test**

```python
from experiments.pravo_nn.corpus_collector.extract import (
    Article,
    extract_articles,
    normalize_whitespace,
    split_articles,
)

# A tiny synthetic raw doc in the spike's chosen format (HTML shown here).
# Replace with a slice of tests/fixtures/sample_raw.txt once the spike lands,
# keeping the asserted invariants below.
RAW_HTML = (
    "<html><body>"
    "<p>ГРАЖДАНСКИЙ КОДЕКС</p>"
    "<p>Статья 1. Основные начала</p>"
    "<p>1. Гражданское законодательство основывается на равенстве.</p>"
    "<p>2</p>"  # standalone page number — must be dropped
    "<p>Статья 2. Регулируемые отношения</p>"
    "<p>Гражданское законодательство определяет правовое положение.</p>"
    "</body></html>"
)


def test_normalize_drops_page_numbers_and_collapses_whitespace():
    out = normalize_whitespace("Статья 1\n\n  много   пробелов \n42\nтекст")
    assert "  " not in out  # runs collapsed
    assert "\n42\n" not in out  # standalone number dropped
    assert "Статья 1" in out and "текст" in out


def test_split_articles_yields_one_article_per_marker():
    text = "преамбула\nСтатья 1\nтело один\nСтатья 2\nтело два"
    arts = split_articles(text, code="ГК РФ", source_url="http://x", date="")
    assert [a.article for a in arts] == ["Статья 1", "Статья 2"]
    assert arts[0].text == "тело один"
    assert all(isinstance(a, Article) and a.code == "ГК РФ" for a in arts)


def test_extract_articles_end_to_end_strips_tags_and_splits():
    arts = extract_articles(RAW_HTML, code="ГК РФ", source_url="http://x", date="1994-11-30")
    assert len(arts) == 2
    assert arts[0].article == "Статья 1. Основные начала"
    assert "равенстве" in arts[0].text
    assert "<p>" not in arts[0].text and "<" not in arts[1].text  # no tags survive
    assert all(a.date == "1994-11-30" for a in arts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_extract.py -v`
Expected: FAIL — `cannot import name 'Article'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Turn a raw fetched document into clean `Article`s.

`strip_to_text` is the only source-specific piece (HTML tag-strip below); the
splitter + normalizer are source-independent. Output is the contract every
downstream consumer depends on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ARTICLE_RE = re.compile(r"(Статья\s+\d+(?:\.\d+)?[^\n]*)")
_PAGE_NUM_RE = re.compile(r"^\d+$")
_WS_RE = re.compile(r"[ \t ]+")
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITIES = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"'}


@dataclass(frozen=True)
class Article:
    code: str
    article: str
    text: str
    source_url: str
    date: str


def normalize_whitespace(text: str) -> str:
    """Collapse intra-line whitespace, drop standalone page-number lines and
    blank lines."""
    out: list[str] = []
    for line in text.splitlines():
        line = _WS_RE.sub(" ", line).strip()
        if not line or _PAGE_NUM_RE.match(line):
            continue
        out.append(line)
    return "\n".join(out)


def strip_to_text(raw: str) -> str:
    """Source-specific shim. HTML source: drop tags + unescape common entities.
    (Plain-text source: return `raw`. JSON source: return the parsed text field.)"""
    text = _TAG_RE.sub("\n", raw)
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    return text


def split_articles(text: str, *, code: str, source_url: str, date: str) -> list[Article]:
    """Split normalized text on `Статья N` markers into Articles (marker = the
    `article` field, the following body = `text`)."""
    parts = _ARTICLE_RE.split(text)
    articles: list[Article] = []
    for i in range(1, len(parts), 2):
        marker = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        articles.append(
            Article(code=code, article=marker, text=body, source_url=source_url, date=date)
        )
    return articles


def extract_articles(raw: str, *, code: str, source_url: str, date: str) -> list[Article]:
    text = normalize_whitespace(strip_to_text(raw))
    return split_articles(text, code=code, source_url=source_url, date=date)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_extract.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/corpus_collector/extract.py experiments/pravo_nn/tests/test_extract.py
git commit -m "feat(pravo-nn): extract clean Articles from raw documents"
```

---

## Task 5: `assemble.py` — write the corpus output contract

**Files:**
- Create: `experiments/pravo_nn/corpus_collector/assemble.py`
- Test: `experiments/pravo_nn/tests/test_assemble.py`

- [ ] **Step 1: Write the failing test**

```python
import json

from experiments.pravo_nn.corpus_collector.assemble import write_corpus
from experiments.pravo_nn.corpus_collector.extract import Article


def _articles():
    return [
        Article("ГК РФ", "Статья 1", "тело один", "http://x", "1994-11-30"),
        Article("ГК РФ", "Статья 2", "тело два", "http://x", "1994-11-30"),
    ]


def test_write_corpus_emits_jsonl_one_object_per_article(tmp_path):
    write_corpus(_articles(), tmp_path)
    lines = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {
        "code": "ГК РФ",
        "article": "Статья 1",
        "text": "тело один",
        "source_url": "http://x",
        "date": "1994-11-30",
    }


def test_write_corpus_emits_concatenated_txt(tmp_path):
    write_corpus(_articles(), tmp_path)
    txt = (tmp_path / "corpus.txt").read_text(encoding="utf-8")
    assert "Статья 1" in txt and "тело один" in txt and "тело два" in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_assemble.py -v`
Expected: FAIL — `cannot import name 'write_corpus'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Write the assembled corpus (`corpus.jsonl` + `corpus.txt`) and the provenance
`manifest.json`. The output files are deterministic (no timestamps) so re-runs
are byte-identical — the manifest's `collected_at` is the only date-varying field
and is passed in, not read from the clock here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from experiments.pravo_nn.corpus_collector.extract import Article

# A code whose total body is below this many chars almost certainly failed to
# extract (e.g. an image-PDF that needs OCR). Flagged in the manifest, not silently shipped.
MIN_CODE_CHARS = 500


def write_corpus(articles: Sequence[Article], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "corpus.jsonl"
    txt = out_dir / "corpus.txt"
    with jsonl.open("w", encoding="utf-8") as jf, txt.open("w", encoding="utf-8") as tf:
        for a in articles:
            jf.write(
                json.dumps(
                    {
                        "code": a.code,
                        "article": a.article,
                        "text": a.text,
                        "source_url": a.source_url,
                        "date": a.date,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            tf.write(f"{a.article}\n{a.text}\n\n")


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_manifest(
    per_code: Mapping[str, Sequence[Article]], *, collected_at: str, source: str
) -> dict:
    documents = []
    for code, arts in per_code.items():
        body = "".join(a.text for a in arts)
        documents.append(
            {
                "code": code,
                "source_url": arts[0].source_url if arts else "",
                "date": arts[0].date if arts else "",
                "articles": len(arts),
                "bytes": len(body.encode("utf-8")),
                "md5": _md5(body),
                "suspiciously_small": len(body) < MIN_CODE_CHARS,
            }
        )
    return {
        "collected_at": collected_at,
        "source": source,
        "documents": documents,
        "total_documents": len(documents),
        "total_bytes": sum(d["bytes"] for d in documents),
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_assemble.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/corpus_collector/assemble.py experiments/pravo_nn/tests/test_assemble.py
git commit -m "feat(pravo-nn): assemble corpus.jsonl + corpus.txt"
```

---

## Task 6: `manifest` integrity + small-doc canary

**Files:**
- Test: `experiments/pravo_nn/tests/test_manifest.py`
- (Implementation already in `assemble.py` from Task 5 — this task tests it and adds nothing new unless a test fails.)

- [ ] **Step 1: Write the failing test**

```python
from experiments.pravo_nn.corpus_collector.assemble import build_manifest
from experiments.pravo_nn.corpus_collector.extract import Article


def test_manifest_counts_and_bytes_match_documents():
    big = "т" * 600  # over MIN_CODE_CHARS
    per_code = {
        "ГК РФ": [Article("ГК РФ", "Статья 1", big, "http://gk", "1994-11-30")],
        "УК РФ": [Article("УК РФ", "Статья 1", big, "http://uk", "1996-06-13")],
    }
    m = build_manifest(per_code, collected_at="2026-06-14", source="mirror")
    assert m["total_documents"] == 2
    assert m["source"] == "mirror"
    assert m["total_bytes"] == sum(d["bytes"] for d in m["documents"])
    assert {d["code"] for d in m["documents"]} == {"ГК РФ", "УК РФ"}
    assert all(len(d["md5"]) == 32 for d in m["documents"])


def test_manifest_flags_suspiciously_small_doc():
    per_code = {"ВзК РФ": [Article("ВзК РФ", "Статья 1", "крошечный", "http://vzk", "")]}
    m = build_manifest(per_code, collected_at="2026-06-14", source="mirror")
    doc = m["documents"][0]
    assert doc["suspiciously_small"] is True  # PDF/OCR canary fired
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_manifest.py -v`
Expected: PASS immediately (the implementation exists from Task 5). If it FAILS, fix `build_manifest` in `assemble.py` until green — do not change the test.

- [ ] **Step 3: Commit**

```bash
git add experiments/pravo_nn/tests/test_manifest.py
git commit -m "test(pravo-nn): manifest integrity + small-doc canary"
```

---

## Task 7: idempotency — re-assembling the same input is a byte no-op

**Files:**
- Test: `experiments/pravo_nn/tests/test_idempotent.py`

- [ ] **Step 1: Write the failing test**

```python
from experiments.pravo_nn.corpus_collector.assemble import write_corpus
from experiments.pravo_nn.corpus_collector.extract import Article


def test_rewriting_same_articles_produces_identical_bytes(tmp_path):
    arts = [
        Article("ГК РФ", "Статья 1", "тело один", "http://x", "1994-11-30"),
        Article("ГК РФ", "Статья 2", "тело два", "http://x", "1994-11-30"),
    ]
    write_corpus(arts, tmp_path)
    first = (tmp_path / "corpus.jsonl").read_bytes()
    write_corpus(arts, tmp_path)  # second run over the same input
    second = (tmp_path / "corpus.jsonl").read_bytes()
    assert first == second  # corpus files carry no timestamps -> deterministic
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_idempotent.py -v`
Expected: PASS (write_corpus is deterministic). If it FAILS, the writer introduced nondeterminism (e.g. dict ordering) — fix `write_corpus`, not the test.

- [ ] **Step 3: Commit**

```bash
git add experiments/pravo_nn/tests/test_idempotent.py
git commit -m "test(pravo-nn): corpus assembly is idempotent"
```

---

## Task 8: `fetch.py` — cached, retrying network layer (network mocked in tests)

Built against the spike's chosen source. `url_for` encodes that source's URL convention; adapt its body to the spike outcome. The cache + retry/backoff + error semantics are source-independent and fully tested with an injected fake opener — **tests never hit the network**.

**Files:**
- Create: `experiments/pravo_nn/corpus_collector/fetch.py`
- Modify: `experiments/pravo_nn/corpus_collector/config.py` (set `SOURCE_BASE` to the spike's value)
- Test: `experiments/pravo_nn/tests/test_fetch.py`

- [ ] **Step 1: Write the failing test**

```python
import urllib.error

import pytest

from experiments.pravo_nn.corpus_collector.config import CodeSpec
from experiments.pravo_nn.corpus_collector.fetch import FetchError, fetch_raw

SPEC = CodeSpec("ГК РФ", "gk-rf")


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_writes_and_returns_text(tmp_path):
    calls = []

    def opener(url):
        calls.append(url)
        return _Resp("Статья 1 текст".encode("utf-8"))

    out = fetch_raw(SPEC, source_base="http://src", cache_dir=tmp_path, opener=opener)
    assert "Статья 1" in out
    assert (tmp_path / "gk-rf.raw").exists()
    assert len(calls) == 1


def test_fetch_uses_cache_without_network(tmp_path):
    (tmp_path / "gk-rf.raw").write_text("cached", encoding="utf-8")

    def opener(url):  # must not be called
        raise AssertionError("network hit despite cache")

    assert fetch_raw(SPEC, source_base="http://src", cache_dir=tmp_path, opener=opener) == "cached"


def test_fetch_retries_then_raises(tmp_path):
    attempts = []

    def opener(url):
        attempts.append(url)
        raise urllib.error.URLError("boom")

    with pytest.raises(FetchError):
        fetch_raw(
            SPEC,
            source_base="http://src",
            cache_dir=tmp_path,
            opener=opener,
            retries=3,
            sleep=lambda _s: None,  # no real waiting in tests
        )
    assert len(attempts) == 3  # all retries exhausted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_fetch.py -v`
Expected: FAIL — `cannot import name 'FetchError'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Network layer: build the source URL for a code, fetch it (once), cache the raw
bytes on disk, retry transient failures with linear backoff. The opener and sleep
are injectable so tests run offline."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from experiments.pravo_nn.corpus_collector.config import CodeSpec


class FetchError(Exception):
    """Raised when a code cannot be fetched after all retries."""


def url_for(spec: CodeSpec, *, source_base: str) -> str:
    """Map a code to its URL at the chosen source. ADAPT to the spike outcome
    (this default assumes `<base>/<slug>`)."""
    return f"{source_base.rstrip('/')}/{spec.slug}"


def fetch_raw(
    spec: CodeSpec,
    *,
    source_base: str,
    cache_dir: Path,
    opener: Callable = urllib.request.urlopen,
    retries: int = 3,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    cache = cache_dir / f"{spec.slug}.raw"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    url = url_for(spec, source_base=source_base)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with opener(url) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(data, encoding="utf-8")
            return data
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                sleep(backoff * (attempt + 1))
    raise FetchError(f"failed to fetch {spec.name} from {url}: {last_exc}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_fetch.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Set the real source base**

Edit `config.py`: set `SOURCE_BASE` to the base URL the spike committed (e.g. the mirror or ИПС base). If `url_for`'s `<base>/<slug>` convention does not match the source, adapt `url_for`'s body (and update `test_fetch` only if the URL *shape* changed, never to weaken the cache/retry assertions).

- [ ] **Step 6: Commit**

```bash
git add experiments/pravo_nn/corpus_collector/fetch.py experiments/pravo_nn/corpus_collector/config.py experiments/pravo_nn/tests/test_fetch.py
git commit -m "feat(pravo-nn): cached retrying fetch layer (network mocked in tests)"
```

---

## Task 9: `cli.py` — the `collect` orchestration command

**Files:**
- Create: `experiments/pravo_nn/corpus_collector/cli.py`
- Test: `experiments/pravo_nn/tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

The CLI is wired so its network + clock are injectable; the test drives the whole pipeline offline with two fake codes.

```python
import json

from experiments.pravo_nn.corpus_collector import cli
from experiments.pravo_nn.corpus_collector.config import CodeSpec


def test_collect_runs_full_pipeline_offline(tmp_path, monkeypatch):
    codes = (CodeSpec("ГК РФ", "gk-rf"), CodeSpec("УК РФ", "uk-rf"))
    raw_by_slug = {
        "gk-rf": "Статья 1\n" + "г" * 600 + "\nСтатья 2\nещё",
        "uk-rf": "Статья 1\n" + "у" * 600,
    }

    def fake_fetch(spec, *, source_base, cache_dir, **kw):
        return raw_by_slug[spec.slug]

    monkeypatch.setattr(cli.fetch, "fetch_raw", fake_fetch)

    cli.collect(
        codes=codes,
        source_base="http://src",
        source_label="mirror",
        data_dir=tmp_path,
        collected_at="2026-06-14",
    )

    corpus = (tmp_path / "corpus" / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(corpus) == 3  # 2 articles in ГК + 1 in УК
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["total_documents"] == 2
    assert manifest["collected_at"] == "2026-06-14"
    assert all(d["suspiciously_small"] is False for d in manifest["documents"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_cli.py -v`
Expected: FAIL — `module ... has no attribute 'collect'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""`collect` command: fetch → extract → assemble for every configured code.

A failed fetch is logged and that code is skipped (partial corpus is still
useful); the gap is visible because the code is simply absent from the manifest.
No silent success."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Sequence

from experiments.pravo_nn.corpus_collector import assemble, config, extract, fetch
from experiments.pravo_nn.corpus_collector.config import CodeSpec
from experiments.pravo_nn.corpus_collector.fetch import FetchError

LOGGER = logging.getLogger(__name__)
_DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"


def collect(
    *,
    codes: Sequence[CodeSpec],
    source_base: str,
    source_label: str,
    data_dir: Path,
    collected_at: str,
) -> None:
    raw_dir = data_dir / "raw"
    corpus_dir = data_dir / "corpus"
    all_articles: list[extract.Article] = []
    per_code: dict[str, list[extract.Article]] = {}
    for spec in codes:
        try:
            raw = fetch.fetch_raw(spec, source_base=source_base, cache_dir=raw_dir)
        except FetchError as exc:
            LOGGER.error("skipping %s: %s", spec.name, exc)
            continue
        url = fetch.url_for(spec, source_base=source_base)
        arts = extract.extract_articles(raw, code=spec.name, source_url=url, date="")
        if not arts:
            LOGGER.warning("%s yielded 0 articles — check the extractor / source", spec.name)
        per_code[spec.name] = arts
        all_articles.extend(arts)

    assemble.write_corpus(all_articles, corpus_dir)
    manifest = assemble.build_manifest(per_code, collected_at=collected_at, source=source_label)
    assemble.write_manifest(manifest, data_dir / "manifest.json")
    small = [d["code"] for d in manifest["documents"] if d["suspiciously_small"]]
    if small:
        LOGGER.warning("suspiciously small (possible image-PDF): %s", ", ".join(small))
    LOGGER.info(
        "wrote %d articles from %d codes to %s",
        len(all_articles),
        len(per_code),
        corpus_dir,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="corpus_collector")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("collect", help="fetch + assemble the RF-codes corpus")
    args = parser.parse_args(argv)
    if args.command == "collect":
        if not config.SOURCE_BASE:
            parser.error("config.SOURCE_BASE is empty — run the Task 3 spike and set it")
        collect(
            codes=config.CODES,
            source_base=config.SOURCE_BASE,
            source_label=config.SOURCE_BASE,
            data_dir=_DEFAULT_DATA,
            collected_at=date.today().isoformat(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_cli.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/corpus_collector/cli.py experiments/pravo_nn/tests/test_cli.py
git commit -m "feat(pravo-nn): collect CLI orchestrating fetch/extract/assemble"
```

---

## Task 10: Full suite green + real collection run + README finalization

**Files:**
- Modify: `experiments/pravo_nn/README.md`

- [ ] **Step 1: Run the whole package test suite**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests -v`
Expected: all tests PASS (config, extract, assemble, manifest, idempotent, fetch, cli).

- [ ] **Step 2: Run the real collection once**

Run: `py -3.13 -m experiments.pravo_nn.corpus_collector.cli collect`
Expected: `data/corpus/corpus.jsonl`, `data/corpus/corpus.txt`, `data/manifest.json` created. Open the manifest: `total_documents` ≈ len(CODES), no `suspiciously_small: true` entries (if any are flagged, the source for that code is likely an image-PDF — revisit the spike for it).

- [ ] **Step 3: Sanity-check the corpus by eye**

Read the first ~30 lines of `data/corpus/corpus.txt`. Confirm it is clean Russian legal text (article markers present, no HTML tags, no page-number litter, no mojibake). Total size should be tens of MB.

- [ ] **Step 4: Finalize the README**

Record the actual outcome: number of codes collected, total bytes, any codes that failed/were flagged, and the confirmed source semantics. State the Definition of Done is met.

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/README.md
git commit -m "docs(pravo-nn): finalize README after first full collection run"
```

---

## Definition of Done (from spec)

- [ ] `py -3 -m experiments.pravo_nn.corpus_collector.cli collect` reproducibly assembles ~20 RF codes into clean Russian text (tens of MB) with a `manifest.json`.
- [ ] All tests green (`py -3.13 -m pytest experiments/pravo_nn/tests`).
- [ ] Text source documented in README per the spike outcome (incl. as-published vs consolidated semantics).
- [ ] No code touches the production tree (`app/`).

## Self-Review notes (author checklist, already applied)

- **Spec coverage:** scaffold/location (Task 1), code list (Task 2), source spike + PDF risk gate (Task 3), extract incl. whitespace/page-number cleanup (Task 4), corpus.jsonl+txt output contract (Task 5), manifest provenance + small-doc canary (Task 6), idempotency (Task 7), fetch with cache+retry+backoff and mocked network (Task 8), CLI + partial-corpus-on-failure (Task 9), DoD verification (Task 10). All spec sections map to a task.
- **Type consistency:** `Article(code, article, text, source_url, date)` defined in Task 4 is used identically in Tasks 5–9; `CodeSpec(name, slug)` from Task 2 used in Tasks 8–9; `build_manifest(per_code, collected_at, source)` / `write_corpus(articles, out_dir)` / `fetch_raw(spec, source_base, cache_dir, opener, retries, backoff, sleep)` signatures match across their defining and calling tasks.
- **Known branch point:** Tasks 3/4/8 carry the only source-dependent code (`strip_to_text`, `url_for`, the committed fixture). Their interfaces are fixed; only the parsing/URL bodies adapt to the spike. This is intentional per the spec, not a placeholder.
