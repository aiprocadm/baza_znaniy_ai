# Mini-GPT v2 — Mixed Law+Wikipedia Warm-Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue training the loss-4.60 mini-GPT by mixing ~12 MB of general Russian Wikipedia text (~50/50 with the legal corpus) and warm-starting from `ckpt.pt`, to teach grammar — the documented data bottleneck.

**Architecture:** Two new sibling packages under `experiments/pravo_nn/` — `wiki_collector/` (Wikimedia API → clean plaintext → bounded sample) and `corpus_mix/` (law+wiki → 50/50 mixed corpus) — plus surgical edits to `mini_gpt/data.py` (val split) and `mini_gpt/train.py` (resume + optimizer state + real val-loss). The vocab-4097 tokenizer is reused unchanged, so the model shape matches the existing checkpoint and weights warm-start directly. The #1↔#2 LoRA checkpoint contract is preserved (only additive keys).

**Tech Stack:** Python 3.13 (`py -3.13`), PyTorch (already installed, no venv), stdlib `urllib`/`json`/`re` for the collector (no new third-party dep), pytest with offline-injected openers.

**Spec:** `docs/superpowers/specs/2026-06-20-mini-gpt-v2-mixed-corpus-warmstart-design.md`

**Conventions for every task:** run tests with `py -3.13 -m pytest` (never bare `py -3` — that is a 3.14 with no torch/pytest). All new code lives under `experiments/pravo_nn/` and must not touch `app/`.

---

## File Structure

| File | Responsibility |
|---|---|
| `experiments/pravo_nn/wiki_collector/__init__.py` | package marker |
| `experiments/pravo_nn/wiki_collector/config.py` | `WikiConfig` dataclass (target bytes, batch size, endpoint, UA) |
| `experiments/pravo_nn/wiki_collector/fetch.py` | one API batch of random-article plaintext; injectable opener, retry/backoff |
| `experiments/pravo_nn/wiki_collector/clean.py` | strip `== headings ==`, collapse blanks, reject stubs |
| `experiments/pravo_nn/wiki_collector/assemble.py` | accumulate to target bytes, dedupe by title, write `wiki.txt` + manifest |
| `experiments/pravo_nn/wiki_collector/cli.py` | `collect_wiki` loop: fetch→clean→accumulate, cache raw batches |
| `experiments/pravo_nn/corpus_mix/__init__.py` | package marker |
| `experiments/pravo_nn/corpus_mix/assemble.py` | mix law+wiki ~50/50 by bytes → `corpus_mixed.txt` + manifest |
| `experiments/pravo_nn/mini_gpt/data.py` | **EDIT**: add `encode_corpus_split` (train/val bins) |
| `experiments/pravo_nn/mini_gpt/train.py` | **EDIT**: optimizer in checkpoint, `resume_from`, `estimate_loss`, val-loss |
| `experiments/pravo_nn/tests/test_wiki_fetch.py` | fetch/parse tests |
| `experiments/pravo_nn/tests/test_wiki_clean.py` | clean tests |
| `experiments/pravo_nn/tests/test_wiki_assemble.py` | accumulate + collect_wiki loop tests |
| `experiments/pravo_nn/tests/test_corpus_mix.py` | mix ratio + manifest tests |
| `experiments/pravo_nn/tests/test_mini_gpt_data_split.py` | val-split tests |
| `experiments/pravo_nn/tests/test_mini_gpt_train_resume.py` | resume/optimizer/val-loss tests |

---

## Task 1: Wikipedia fetch — build URL + parse batch JSON

**Files:**
- Create: `experiments/pravo_nn/wiki_collector/__init__.py`
- Create: `experiments/pravo_nn/wiki_collector/fetch.py`
- Test: `experiments/pravo_nn/tests/test_wiki_fetch.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/pravo_nn/tests/test_wiki_fetch.py
import json
import urllib.error

import pytest

from experiments.pravo_nn.wiki_collector.fetch import (
    WikiFetchError,
    batch_url,
    fetch_batch,
    parse_batch,
)

_PAYLOAD = json.dumps(
    {
        "batchcomplete": "",
        "query": {
            "pages": {
                "12": {"pageid": 12, "ns": 0, "title": "Пушкин", "extract": "Поэт.\n== Жизнь ==\nРодился."},
                "34": {"pageid": 34, "ns": 0, "title": "Стуб", "extract": ""},
            }
        },
    }
)


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_batch_url_has_random_plaintext_params():
    url = batch_url(api_url="https://ru.wikipedia.org/w/api.php", limit=20)
    assert "generator=random" in url
    assert "explaintext=1" in url
    assert "grnlimit=20" in url
    assert "prop=extracts" in url


def test_parse_batch_keeps_titled_nonempty_extracts():
    pairs = parse_batch(_PAYLOAD)
    assert ("Пушкин", "Поэт.\n== Жизнь ==\nРодился.") in pairs
    assert all(title != "Стуб" for title, _ in pairs)  # empty extract dropped


def test_fetch_batch_offline_via_injected_opener():
    calls = []

    def opener(req):
        calls.append(req)
        return _Resp(_PAYLOAD.encode("utf-8"))

    pairs = fetch_batch(opener=opener, limit=20)
    assert ("Пушкин", "Поэт.\n== Жизнь ==\nРодился.") in pairs
    assert len(calls) == 1


def test_fetch_batch_retries_then_raises():
    attempts = []

    def opener(req):
        attempts.append(req)
        raise urllib.error.URLError("boom")

    with pytest.raises(WikiFetchError):
        fetch_batch(opener=opener, retries=3, sleep=lambda _s: None)
    assert len(attempts) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_wiki_fetch.py -v`
Expected: FAIL with `ModuleNotFoundError: experiments.pravo_nn.wiki_collector.fetch`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/pravo_nn/wiki_collector/__init__.py
```
(empty file)

```python
# experiments/pravo_nn/wiki_collector/fetch.py
"""Network layer for the Wikipedia sample: build the API URL and fetch ONE batch
of random-article plaintext extracts. The opener and sleep are injectable so
tests run fully offline (mirrors corpus_collector/fetch.py)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

API_URL = "https://ru.wikipedia.org/w/api.php"
USER_AGENT = "pravo-nn-research/1.0 (mini-GPT corpus; aiproc.adm@gmail.com)"


class WikiFetchError(Exception):
    """Raised when a batch cannot be fetched after all retries."""


def batch_url(*, api_url: str, limit: int) -> str:
    """One request that yields `limit` random article plaintext extracts."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "wiki",  # keep "== Heading ==" markers for clean.py to strip
        "exlimit": "max",
        "generator": "random",
        "grnnamespace": "0",  # article namespace only
        "grnlimit": str(limit),
    }
    return api_url + "?" + urllib.parse.urlencode(params)


def parse_batch(payload: str) -> list[tuple[str, str]]:
    """(title, plaintext) pairs from one API JSON response; drops empty extracts."""
    data = json.loads(payload)
    pages = data.get("query", {}).get("pages", {})
    out: list[tuple[str, str]] = []
    for page in pages.values():
        title = page.get("title", "")
        extract = page.get("extract", "")
        if title and extract:
            out.append((title, extract))
    return out


def fetch_batch(
    *,
    api_url: str = API_URL,
    limit: int = 20,
    opener: Callable = urllib.request.urlopen,
    retries: int = 3,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    user_agent: str = USER_AGENT,
) -> list[tuple[str, str]]:
    url = batch_url(api_url=api_url, limit=limit)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with opener(req) as resp:
                payload = resp.read().decode("utf-8", errors="replace")
            return parse_batch(payload)
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                sleep(backoff * (attempt + 1))
    raise WikiFetchError(f"failed to fetch wiki batch from {url}: {last_exc}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_wiki_fetch.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/wiki_collector/__init__.py experiments/pravo_nn/wiki_collector/fetch.py experiments/pravo_nn/tests/test_wiki_fetch.py
git commit -m "feat(wiki): Wikimedia API batch fetch for the mini-GPT v2 corpus"
```

---

## Task 2: Wikipedia clean — strip headings, drop stubs

**Files:**
- Create: `experiments/pravo_nn/wiki_collector/clean.py`
- Test: `experiments/pravo_nn/tests/test_wiki_clean.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/pravo_nn/tests/test_wiki_clean.py
from experiments.pravo_nn.wiki_collector.clean import clean_extract, is_substantial


def test_clean_strips_section_headings():
    raw = "Первый абзац.\n== История ==\nВторой абзац.\n=== Подраздел ===\nТретий."
    out = clean_extract(raw)
    assert "==" not in out
    assert "История" not in out  # the heading line is removed whole
    assert "Первый абзац." in out
    assert "Второй абзац." in out


def test_clean_collapses_blank_runs():
    out = clean_extract("А.\n\n\n\nБ.")
    assert "\n\n\n" not in out
    assert out == "А.\n\nБ."


def test_is_substantial_rejects_short_stub():
    assert not is_substantial("Слишком коротко.")
    assert is_substantial("длинный текст " * 50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_wiki_clean.py -v`
Expected: FAIL with `ModuleNotFoundError: ...wiki_collector.clean`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/pravo_nn/wiki_collector/clean.py
"""Turn a raw Wikipedia plaintext extract into training text: remove section
heading lines ("== История =="), collapse blank runs, and judge whether an
article carries enough prose to be worth keeping (stubs add noise, not grammar)."""

from __future__ import annotations

import re

# A whole line that is just a "== ... ==" / "=== ... ===" heading.
_HEADING_RE = re.compile(r"^\s*={2,}.*?={2,}\s*$", re.MULTILINE)
_BLANKS_RE = re.compile(r"\n{3,}")
MIN_ARTICLE_CHARS = 200


def clean_extract(text: str) -> str:
    text = _HEADING_RE.sub("", text)
    text = _BLANKS_RE.sub("\n\n", text)
    return text.strip()


def is_substantial(text: str, *, min_chars: int = MIN_ARTICLE_CHARS) -> bool:
    return len(text) >= min_chars
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_wiki_clean.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/wiki_collector/clean.py experiments/pravo_nn/tests/test_wiki_clean.py
git commit -m "feat(wiki): clean plaintext extracts (strip headings, drop stubs)"
```

---

## Task 3: Wikipedia assemble — accumulate to target, dedupe, manifest

**Files:**
- Create: `experiments/pravo_nn/wiki_collector/assemble.py`
- Test: `experiments/pravo_nn/tests/test_wiki_assemble.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/pravo_nn/tests/test_wiki_assemble.py
from experiments.pravo_nn.wiki_collector.assemble import (
    accumulate,
    build_manifest,
    write_wiki,
)


def test_accumulate_stops_at_target_and_dedupes():
    arts = [
        ("A", "x" * 100),
        ("A", "x" * 100),   # duplicate title — skipped
        ("B", "y" * 100),
        ("C", "z" * 100),   # should not be reached once target hit at B
    ]
    kept, total = accumulate(iter(arts), target_bytes=150)
    titles = [t for t, _ in kept]
    assert titles == ["A", "B"]      # dedup + stopped after crossing 150 bytes
    assert total >= 150


def test_write_wiki_and_manifest(tmp_path):
    kept = [("A", "альфа текст"), ("B", "бета текст")]
    write_wiki(kept, tmp_path)
    assert (tmp_path / "wiki.txt").exists()
    body = (tmp_path / "wiki.txt").read_text(encoding="utf-8")
    assert "альфа текст" in body and "бета текст" in body

    manifest = build_manifest(kept, collected_at="2026-06-20", source="https://ru.wikipedia.org")
    assert manifest["articles"] == 2
    assert manifest["titles"] == ["A", "B"]
    assert manifest["total_bytes"] == len("альфа текст".encode()) + len("бета текст".encode())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_wiki_assemble.py -v`
Expected: FAIL with `ModuleNotFoundError: ...wiki_collector.assemble`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/pravo_nn/wiki_collector/assemble.py
"""Accumulate cleaned Wikipedia articles up to a byte budget, dedupe by title,
and write wiki.txt + a provenance manifest (deterministic — no clock reads)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def accumulate(
    articles: Iterable[tuple[str, str]],
    *,
    target_bytes: int,
) -> tuple[list[tuple[str, str]], int]:
    kept: list[tuple[str, str]] = []
    seen: set[str] = set()
    total = 0
    for title, text in articles:
        if title in seen:
            continue
        seen.add(title)
        kept.append((title, text))
        total += len(text.encode("utf-8"))
        if total >= target_bytes:
            break
    return kept, total


def write_wiki(kept: list[tuple[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "wiki.txt").open("w", encoding="utf-8") as tf:
        for _title, text in kept:
            tf.write(text + "\n\n")


def build_manifest(kept: list[tuple[str, str]], *, collected_at: str, source: str) -> dict:
    return {
        "collected_at": collected_at,
        "source": source,
        "articles": len(kept),
        "total_bytes": sum(len(x.encode("utf-8")) for _, x in kept),
        "titles": [t for t, _ in kept],
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_wiki_assemble.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/wiki_collector/assemble.py experiments/pravo_nn/tests/test_wiki_assemble.py
git commit -m "feat(wiki): accumulate to byte budget, dedupe, write manifest"
```

---

## Task 4: Wikipedia config + collect loop (CLI)

**Files:**
- Create: `experiments/pravo_nn/wiki_collector/config.py`
- Create: `experiments/pravo_nn/wiki_collector/cli.py`
- Test: `experiments/pravo_nn/tests/test_wiki_assemble.py` (append a loop test)

- [ ] **Step 1: Write the failing test (append to test_wiki_assemble.py)**

```python
# append to experiments/pravo_nn/tests/test_wiki_assemble.py
from experiments.pravo_nn.wiki_collector.cli import collect_wiki
from experiments.pravo_nn.wiki_collector.config import WikiConfig


def test_collect_wiki_loops_until_target(tmp_path):
    # Each fake batch returns two fresh articles; loop must stop once target met.
    batches = [
        [("A" + str(i), "длинный русский текст " * 30), ("B" + str(i), "ещё текст " * 30)]
        for i in range(50)
    ]
    seq = iter(batches)

    def fake_fetch(*, api_url, limit, user_agent):
        return next(seq)

    cfg = WikiConfig(target_bytes=4000, batch_limit=20)
    out = collect_wiki(
        cfg=cfg,
        data_dir=tmp_path,
        collected_at="2026-06-20",
        fetch=fake_fetch,
        sleep=lambda _s: None,
    )
    assert out.exists()
    assert (tmp_path / "wiki" / "manifest.json").exists()
    body = out.read_text(encoding="utf-8")
    assert len(body.encode("utf-8")) >= 4000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_wiki_assemble.py::test_collect_wiki_loops_until_target -v`
Expected: FAIL with `ModuleNotFoundError: ...wiki_collector.cli`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/pravo_nn/wiki_collector/config.py
"""Knobs for the Wikipedia sample. target_bytes ~12 MB matches the legal corpus
(~12.1 MB on disk) so a 50/50 mix keeps all the law."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WikiConfig:
    target_bytes: int = 12_000_000
    batch_limit: int = 20  # articles per API call (exlimit cap for full extracts)
    api_url: str = "https://ru.wikipedia.org/w/api.php"
    user_agent: str = "pravo-nn-research/1.0 (mini-GPT corpus; aiproc.adm@gmail.com)"
```

```python
# experiments/pravo_nn/wiki_collector/cli.py
"""`collect_wiki`: loop fetch → clean → accumulate until the byte budget is met,
caching each raw batch so a re-run is free and offline. The fetch and sleep are
injectable so tests drive the loop without network."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Callable

from experiments.pravo_nn.wiki_collector import assemble, clean, fetch
from experiments.pravo_nn.wiki_collector.config import WikiConfig

LOGGER = logging.getLogger(__name__)
_DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"


def collect_wiki(
    *,
    cfg: WikiConfig,
    data_dir: Path,
    collected_at: str,
    fetch: Callable = fetch.fetch_batch,
    sleep: Callable[[float], None] = time.sleep,
    max_batches: int = 10000,
) -> Path:
    wiki_dir = data_dir / "wiki"
    raw_dir = wiki_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    kept: list[tuple[str, str]] = []
    seen: set[str] = set()
    total = 0
    batch_no = 0
    while total < cfg.target_bytes and batch_no < max_batches:
        batch_no += 1
        cache = raw_dir / f"batch-{batch_no:04d}.json"
        if cache.exists():
            raw_pairs = [tuple(p) for p in json.loads(cache.read_text(encoding="utf-8"))]
        else:
            raw_pairs = fetch(api_url=cfg.api_url, limit=cfg.batch_limit, user_agent=cfg.user_agent)
            cache.write_text(json.dumps(raw_pairs, ensure_ascii=False), encoding="utf-8")
            sleep(0.5)  # be polite to the API
        for title, extract in raw_pairs:
            if title in seen:
                continue
            cleaned = clean.clean_extract(extract)
            if not clean.is_substantial(cleaned):
                continue
            seen.add(title)
            kept.append((title, cleaned))
            total += len(cleaned.encode("utf-8"))
            if total >= cfg.target_bytes:
                break

    assemble.write_wiki(kept, wiki_dir)
    manifest = assemble.build_manifest(kept, collected_at=collected_at, source=cfg.api_url)
    assemble.write_manifest(manifest, wiki_dir / "manifest.json")
    LOGGER.info("wrote %d wiki articles (%d bytes) to %s", len(kept), total, wiki_dir)
    return wiki_dir / "wiki.txt"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="wiki_collector")
    parser.add_argument("--target-bytes", type=int, default=WikiConfig.target_bytes)
    args = parser.parse_args(argv)
    collect_wiki(
        cfg=WikiConfig(target_bytes=args.target_bytes),
        data_dir=_DEFAULT_DATA,
        collected_at=date.today().isoformat(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: the `fetch` parameter shadows the imported `fetch` module inside `collect_wiki`; that is intentional and safe because the module is only referenced via the default value, evaluated at def-time.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_wiki_assemble.py -v`
Expected: PASS (3 tests total in file)

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/wiki_collector/config.py experiments/pravo_nn/wiki_collector/cli.py experiments/pravo_nn/tests/test_wiki_assemble.py
git commit -m "feat(wiki): collect loop + CLI with raw-batch caching"
```

---

## Task 5: Mix the corpora ~50/50 by bytes

**Files:**
- Create: `experiments/pravo_nn/corpus_mix/__init__.py`
- Create: `experiments/pravo_nn/corpus_mix/assemble.py`
- Test: `experiments/pravo_nn/tests/test_corpus_mix.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/pravo_nn/tests/test_corpus_mix.py
from experiments.pravo_nn.corpus_mix.assemble import mix_corpora


def test_mix_is_balanced_by_bytes():
    law = "закон " * 1000      # large
    wiki = "статья " * 100     # small
    mixed, manifest = mix_corpora(law, wiki)
    # the larger source is truncated to the smaller's budget -> roughly equal
    assert abs(manifest["law_bytes"] - manifest["wiki_bytes"]) <= len("закон ".encode())
    assert "закон" in mixed and "статья" in mixed


def test_mix_keeps_all_when_already_equal():
    law = "ё" * 100
    wiki = "я" * 100
    mixed, manifest = mix_corpora(law, wiki)
    assert manifest["law_bytes"] == len(law.encode("utf-8"))
    assert manifest["wiki_bytes"] == len(wiki.encode("utf-8"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_corpus_mix.py -v`
Expected: FAIL with `ModuleNotFoundError: ...corpus_mix.assemble`

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/pravo_nn/corpus_mix/__init__.py
```
(empty file)

```python
# experiments/pravo_nn/corpus_mix/assemble.py
"""Mix the legal corpus and the Wikipedia sample ~50/50 by bytes into one
training file. The larger source is truncated (at a newline boundary, so no
article is cut mid-sentence) to the smaller source's byte budget."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    enc = text.encode("utf-8")
    if len(enc) <= max_bytes:
        return text
    cut = enc[:max_bytes].decode("utf-8", errors="ignore")
    nl = cut.rfind("\n")
    return cut[:nl] if nl > 0 else cut


def mix_corpora(law_text: str, wiki_text: str) -> tuple[str, dict]:
    budget = min(len(law_text.encode("utf-8")), len(wiki_text.encode("utf-8")))
    law_keep = _truncate_to_bytes(law_text, budget)
    wiki_keep = _truncate_to_bytes(wiki_text, budget)
    mixed = law_keep + "\n\n" + wiki_keep
    manifest = {
        "law_bytes": len(law_keep.encode("utf-8")),
        "wiki_bytes": len(wiki_keep.encode("utf-8")),
        "law_md5": hashlib.md5(law_keep.encode("utf-8")).hexdigest(),
        "wiki_md5": hashlib.md5(wiki_keep.encode("utf-8")).hexdigest(),
        "ratio": "50/50",
    }
    return mixed, manifest


def write_mixed(mixed: str, manifest: dict, *, out_path: Path, manifest_path: Path, collected_at: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mixed, encoding="utf-8")
    manifest_path.write_text(
        json.dumps({**manifest, "collected_at": collected_at}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_corpus_mix.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/corpus_mix/__init__.py experiments/pravo_nn/corpus_mix/assemble.py experiments/pravo_nn/tests/test_corpus_mix.py
git commit -m "feat(corpus-mix): assemble law+wiki 50/50 by bytes with manifest"
```

---

## Task 6: Data — encode with a held-out val split

**Files:**
- Modify: `experiments/pravo_nn/mini_gpt/data.py` (add `encode_corpus_split`)
- Test: `experiments/pravo_nn/tests/test_mini_gpt_data_split.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/pravo_nn/tests/test_mini_gpt_data_split.py
from experiments.pravo_nn.mini_gpt.data import encode_corpus_split, load_bin
from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer

SAMPLE = "Статья 1. Основные начала.\nСтатья 2. Регулируемые отношения.\n" * 50


def _tok():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    return tok


def test_encode_split_writes_disjoint_train_and_val(tmp_path):
    tok = _tok()
    n_train, n_val = encode_corpus_split(
        SAMPLE, tok,
        train_path=tmp_path / "train.bin",
        val_path=tmp_path / "val.bin",
        val_frac=0.1,
    )
    train = load_bin(tmp_path / "train.bin")
    val = load_bin(tmp_path / "val.bin")
    assert len(train) == n_train and len(val) == n_val
    assert n_val > 0
    # val is the tail of the full token stream — train ends where val begins
    full = list(train) + list(val)
    assert len(full) == n_train + n_val
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_data_split.py -v`
Expected: FAIL with `ImportError: cannot import name 'encode_corpus_split'`

- [ ] **Step 3: Write minimal implementation (append to data.py)**

```python
# append to experiments/pravo_nn/mini_gpt/data.py
def encode_corpus_split(
    text: str,
    tokenizer: BPETokenizer,
    *,
    train_path,
    val_path,
    val_frac: float = 0.05,
) -> tuple[int, int]:
    """Encode once, then reserve the LAST `val_frac` of tokens as a held-out
    val.bin. Returns (n_train, n_val)."""
    ids = tokenizer.encode(text)
    if ids and max(ids) > 65535:
        raise ValueError(f"token id {max(ids)} exceeds uint16; vocab too large for .bin")
    arr = np.array(ids, dtype=np.uint16)
    n_val = int(len(arr) * val_frac)
    split = len(arr) - n_val
    train_arr, val_arr = arr[:split], arr[split:]
    for out_path, chunk in ((train_path, train_arr), (val_path, val_arr)):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        chunk.tofile(out)
    return len(train_arr), len(val_arr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_data_split.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add experiments/pravo_nn/mini_gpt/data.py experiments/pravo_nn/tests/test_mini_gpt_data_split.py
git commit -m "feat(mini-gpt): encode_corpus_split with held-out val tail"
```

---

## Task 7: Train — optimizer in checkpoint, resume, real val-loss

**Files:**
- Modify: `experiments/pravo_nn/mini_gpt/train.py`
- Test: `experiments/pravo_nn/tests/test_mini_gpt_train_resume.py`

- [ ] **Step 1: Write the failing test**

```python
# experiments/pravo_nn/tests/test_mini_gpt_train_resume.py
import pytest
import torch

from experiments.pravo_nn.mini_gpt.config import GPTConfig
from experiments.pravo_nn.mini_gpt.data import encode_corpus_split
from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer
from experiments.pravo_nn.mini_gpt.train import train

SAMPLE = "Статья 1. Основные начала регулирования отношений в обществе.\n" * 80
# Tiny model so the whole resume cycle runs in well under a second.
TINY = GPTConfig(vocab_size=0, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0)


def _setup(tmp_path):
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    tok.save(tmp_path / "tokenizer")
    encode_corpus_split(
        SAMPLE, tok,
        train_path=tmp_path / "train.bin",
        val_path=tmp_path / "val.bin",
        val_frac=0.1,
    )


def _ckpt(tmp_path):
    return tmp_path / "checkpoints" / "ckpt.pt"


def test_checkpoint_carries_optimizer_and_real_val_loss(tmp_path):
    _setup(tmp_path)
    train(preset=TINY, data_dir=tmp_path, max_steps=3, batch_size=2,
          warmup=1, log_interval=1, ckpt_interval=2, eval_interval=2, eval_iters=2)
    ckpt = torch.load(_ckpt(tmp_path), map_location="cpu", weights_only=False)
    assert ckpt["step"] == 3
    assert "optimizer_state_dict" in ckpt
    assert isinstance(ckpt["val_loss"], float)


def test_resume_continues_step_counter(tmp_path):
    _setup(tmp_path)
    train(preset=TINY, data_dir=tmp_path, max_steps=3, batch_size=2, warmup=1)
    train(preset=TINY, data_dir=tmp_path, max_steps=2, batch_size=2, warmup=1,
          resume_from=_ckpt(tmp_path))
    ckpt = torch.load(_ckpt(tmp_path), map_location="cpu", weights_only=False)
    assert ckpt["step"] == 5  # 3 + 2, not reset to 2


def test_resume_works_without_optimizer_state(tmp_path):
    """Backward compat: the original #1 ckpt_v1 has no optimizer_state_dict."""
    _setup(tmp_path)
    train(preset=TINY, data_dir=tmp_path, max_steps=2, batch_size=2, warmup=1)
    p = _ckpt(tmp_path)
    c = torch.load(p, map_location="cpu", weights_only=False)
    c.pop("optimizer_state_dict")
    torch.save(c, p)
    train(preset=TINY, data_dir=tmp_path, max_steps=1, batch_size=2, warmup=1, resume_from=p)  # must not raise


def test_resume_rejects_vocab_mismatch(tmp_path):
    _setup(tmp_path)
    train(preset=TINY, data_dir=tmp_path, max_steps=2, batch_size=2, warmup=1)
    # retrain the tokenizer to a different vocab, overwriting the dir
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=320)
    tok.save(tmp_path / "tokenizer")
    with pytest.raises(ValueError):
        train(preset=TINY, data_dir=tmp_path, max_steps=1, batch_size=2, warmup=1,
              resume_from=_ckpt(tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_train_resume.py -v`
Expected: FAIL — `train()` has no `resume_from`/`eval_interval` kwargs (TypeError) and the checkpoint has no `optimizer_state_dict`.

- [ ] **Step 3: Write the implementation — replace `save_checkpoint` and `train`, add `estimate_loss`**

Replace `save_checkpoint` (train.py:30-42) with:

```python
def save_checkpoint(model, cfg: GPTConfig, *, tokenizer_dir: str, step: int, val_loss: float, path, optimizer=None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "tokenizer": tokenizer_dir,
        "step": step,
        "val_loss": val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, out)
```

Add `estimate_loss` after `_lr_at` (train.py:52). Note: `model.train(False)` is the
explicit form of inference mode (equivalent to `.eval()`); `.train(True)` restores
training mode afterwards:

```python
@torch.no_grad()
def estimate_loss(model, data, *, block_size: int, batch_size: int, device: str, eval_iters: int = 20) -> float:
    model.train(False)  # inference mode (disables dropout) for a clean val measurement
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size=block_size, batch_size=batch_size, device=device)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train(True)  # back to training mode
    return sum(losses) / len(losses)
```

Replace the entire `train(...)` function (train.py:54-98) with:

```python
def train(
    *,
    preset: GPTConfig = CPU_OVERNIGHT,
    data_dir: Path = _DATA,
    max_steps: int = 5000,
    batch_size: int = 32,
    base_lr: float = 3e-4,
    warmup: int = 100,
    log_interval: int = 250,
    ckpt_interval: int = 500,
    eval_interval: int = 500,
    eval_iters: int = 20,
    resume_from: Path | None = None,
) -> Path:
    device = get_device()
    tok = BPETokenizer.load(data_dir / "tokenizer")
    vocab_size = len(tok.vocab) + len(tok.special_tokens)
    ckpt_path = data_dir / "checkpoints" / "ckpt.pt"

    start_step = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        if ckpt["config"]["vocab_size"] != vocab_size:
            raise ValueError(
                f"tokenizer vocab {vocab_size} != checkpoint vocab {ckpt['config']['vocab_size']}; "
                "warm-start needs the SAME tokenizer (reuse data/tokenizer, do not retrain it)"
            )
        cfg = GPTConfig(**ckpt["config"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"])
        LOGGER.info("resumed from %s at step %d", resume_from, start_step)
    else:
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=preset.block_size,
            n_layer=preset.n_layer,
            n_head=preset.n_head,
            n_embd=preset.n_embd,
            dropout=preset.dropout,
        )
        model = GPT(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=base_lr)

    n_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(
        "device=%s params=%.2fM block=%d vocab=%d start_step=%d",
        device, n_params / 1e6, cfg.block_size, cfg.vocab_size, start_step,
    )

    data = load_bin(data_dir / "train.bin")
    val_path = data_dir / "val.bin"
    val_data = load_bin(val_path) if val_path.exists() else None

    last_loss = float("inf")
    last_val = float("inf")
    for local in range(max_steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(local, base_lr=base_lr, warmup=warmup, total=max_steps, min_lr=base_lr / 10)
        x, y = get_batch(data, block_size=cfg.block_size, batch_size=batch_size, device=device)
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = loss.item()
        abs_step = start_step + local
        if local % log_interval == 0:
            LOGGER.info("step %d (local %d/%d) loss %.4f", abs_step, local, max_steps, last_loss)
        if val_data is not None and local > 0 and local % eval_interval == 0:
            last_val = estimate_loss(model, val_data, block_size=cfg.block_size, batch_size=batch_size, device=device, eval_iters=eval_iters)
            LOGGER.info("step %d val_loss %.4f", abs_step, last_val)
        if local > 0 and local % ckpt_interval == 0:
            save_checkpoint(
                model, cfg, tokenizer_dir="data/tokenizer", step=abs_step,
                val_loss=(last_val if val_data is not None else last_loss),
                path=ckpt_path, optimizer=opt,
            )

    final_val = (
        estimate_loss(model, val_data, block_size=cfg.block_size, batch_size=batch_size, device=device, eval_iters=eval_iters)
        if val_data is not None else last_loss
    )
    save_checkpoint(
        model, cfg, tokenizer_dir="data/tokenizer", step=start_step + max_steps,
        val_loss=final_val, path=ckpt_path, optimizer=opt,
    )
    LOGGER.info("done; final train-loss %.4f val-loss %.4f -> %s", last_loss, final_val, ckpt_path)
    return ckpt_path
```

Replace `main(...)` (train.py:101-110) with:

```python
def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="mini_gpt.train")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-interval", type=int, default=500)
    default_ckpt = str(_DATA / "checkpoints" / "ckpt.pt")
    p.add_argument("--resume", nargs="?", const=default_ckpt, default=None,
                   help="resume (warm-start) from a checkpoint; bare flag uses the default ckpt path")
    args = p.parse_args(argv)
    train(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        resume_from=Path(args.resume) if args.resume else None,
    )
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests/test_mini_gpt_train_resume.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the existing mini-GPT suite for no regressions**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests -k mini_gpt -v`
Expected: PASS (all prior tests + the new ones; the #1↔#2 checkpoint-contract test still passes because new keys are additive)

- [ ] **Step 6: Commit**

```bash
git add experiments/pravo_nn/mini_gpt/train.py experiments/pravo_nn/tests/test_mini_gpt_train_resume.py
git commit -m "feat(mini-gpt): warm-start resume + optimizer state + real val-loss"
```

---

## Task 8: Runbook — collect, mix, re-encode, warm-start (operational, not pytest)

This task runs the real pipeline. It is a documented runbook, not unit tests. Follow the ops rules from repo memory: **`py -3.13` only, exactly one torch process, detached long run with FileHandler logging, back up the checkpoint first.**

- [ ] **Step 1: Full fast suite green before any long run**

Run: `py -3.13 -m pytest experiments/pravo_nn/tests -q`
Expected: all pass.

- [ ] **Step 2: Back up the reproducible #1 artifacts**

```bash
cd experiments/pravo_nn/data
cp checkpoints/ckpt.pt checkpoints/ckpt_v1.pt        # the loss-4.60 base — never lose it
cp train.bin train_legal.bin                          # legal-only bin
cd -
```
Expected: `ckpt_v1.pt` and `train_legal.bin` exist.

- [ ] **Step 3: Collect the Wikipedia sample (live network, ~minutes)**

Run: `py -3.13 -m experiments.pravo_nn.wiki_collector.cli --target-bytes 12000000`
Expected: `experiments/pravo_nn/data/wiki/wiki.txt` ≈ 12 MB + `wiki/manifest.json` with the article titles. Re-running is free (raw batches cached under `data/wiki/raw/`).

- [ ] **Step 4: Assemble the mixed corpus**

```bash
py -3.13 -c "from pathlib import Path; from datetime import date; from experiments.pravo_nn.corpus_mix.assemble import mix_corpora, write_mixed; d=Path('experiments/pravo_nn/data'); law=(d/'corpus'/'corpus.txt').read_text(encoding='utf-8'); wiki=(d/'wiki'/'wiki.txt').read_text(encoding='utf-8'); mixed,man=mix_corpora(law,wiki); write_mixed(mixed,man,out_path=d/'corpus_mixed.txt',manifest_path=d/'corpus_mixed.manifest.json',collected_at=date.today().isoformat()); print(man)"
```
Expected: `corpus_mixed.txt` written; printed manifest shows `law_bytes ≈ wiki_bytes` (~12 MB each).

- [ ] **Step 5: Re-encode the mixed corpus with the EXISTING tokenizer, with val split**

```bash
py -3.13 -c "from pathlib import Path; from experiments.pravo_nn.mini_gpt.tokenizer import BPETokenizer; from experiments.pravo_nn.mini_gpt.data import encode_corpus_split; d=Path('experiments/pravo_nn/data'); tok=BPETokenizer.load(d/'tokenizer'); text=(d/'corpus_mixed.txt').read_text(encoding='utf-8'); ntr,nva=encode_corpus_split(text,tok,train_path=d/'train.bin',val_path=d/'val.bin',val_frac=0.05); print('train',ntr,'val',nva)"
```
Expected: prints token counts; `train.bin` (overwritten) + new `val.bin` written. Vocab is unchanged (4097), so the checkpoint stays compatible.

- [ ] **Step 6: Launch the warm-start run detached, logging via FileHandler**

Create `experiments/pravo_nn/mini_gpt/run_warmstart.py`:

```python
"""Detached warm-start entrypoint: logs to a file via FileHandler (flushes per
record — Start-Process -RedirectStandardError buffers and looks frozen)."""

from __future__ import annotations

import logging
from pathlib import Path

from experiments.pravo_nn.mini_gpt.train import train

_DATA = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    log = _DATA / "checkpoints" / "warmstart_run.log"
    handler = logging.FileHandler(log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.getLogger(__name__).info("RUN START (warm-start on mixed corpus)")
    train(
        max_steps=4000,
        batch_size=8,
        eval_interval=250,
        resume_from=_DATA / "checkpoints" / "ckpt_v1.pt",
    )
    logging.getLogger(__name__).info("RUN COMPLETE")
```

Launch detached (PowerShell), exactly ONE torch process:

```powershell
Start-Process -FilePath py -ArgumentList '-3.13','-m','experiments.pravo_nn.mini_gpt.run_warmstart' -WorkingDirectory (Get-Location) -WindowStyle Hidden
```
Expected: process starts; `data/checkpoints/warmstart_run.log` begins filling with `step … loss …` / `val_loss …` lines.

- [ ] **Step 7: Monitor with a dead-man rule**

Tail the log periodically (harness background tasks die ~10 min into CPU LLM runs — monitor, don't block):

```bash
tail -n 15 experiments/pravo_nn/data/checkpoints/warmstart_run.log
```
Expected over time: train-loss settles after the initial distribution-shift bump; **val_loss trend is the signal of interest**. Checkpoints land every 500 steps; `step` is absolute (resumes past 2000).

- [ ] **Step 8: Sanity-check generation after the run**

```bash
py -3.13 -m experiments.pravo_nn.mini_gpt.generate --prompt "Статья 1." --max-new-tokens 200 --temperature 0.8 --top-k 40
```
Expected: qualitatively more connected Russian than the loss-4.60 word-salad samples (judge by eye; absolute loss may stay near 4.6 due to the mixed distribution).

- [ ] **Step 9: Commit the runbook entrypoint + manifests (NOT the large bins/checkpoints)**

```bash
git add experiments/pravo_nn/mini_gpt/run_warmstart.py
git add experiments/pravo_nn/data/wiki/manifest.json experiments/pravo_nn/data/corpus_mixed.manifest.json
git commit -m "chore(mini-gpt): warm-start runbook entrypoint + corpus provenance manifests"
```
Note: confirm `.gitignore` already excludes `*.bin`, `*.pt`, and `data/wiki/raw/`. If not, add them before committing so the ~24 MB corpus and checkpoints are not committed (matches how #1 kept large artifacts out of git).

---

## Self-Review notes (done by the planner)

- **Spec coverage:** wiki collector (Tasks 1–4) ✓; 50/50 mix (Task 5) ✓; val split (Task 6) ✓; resume + optimizer state + val-loss + vocab guard (Task 7) ✓; backward-compat with optimizer-less ckpt (Task 7 test) ✓; #1↔#2 contract preserved (Task 7 Step 5 runs the existing checkpoint test) ✓; ops rules — `py -3.13`, one process, FileHandler, ckpt backup (Task 8) ✓.
- **Type consistency:** `fetch_batch(api_url=, limit=, user_agent=)` is called with exactly those kwargs by `collect_wiki` and the test's `fake_fetch`. `encode_corpus_split(train_path=, val_path=, val_frac=)` matches across Task 6 and Task 8. `save_checkpoint(..., optimizer=)` and `train(resume_from=, eval_interval=, eval_iters=)` signatures match their callers and tests.
- **Placeholder scan:** no TBD/TODO; every code step is complete.
- **Known live-network caveat:** Task 8 Step 3 hits the real Wikimedia API — it is intentionally outside pytest (mirrors how #0 kept the network out of the test suite).
