# MVP Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Закрыть оставшиеся пункты MVP из vision'а (UI brand polish, one-click installer, backup/restore CLI) и подготовить проект к OSS-публикации под Apache-2.0 с сохранением optionality для будущего pivot в SaaS.

**Architecture:** Три независимых sprint'а (~12ч + ~10ч + ~7.5ч). Каждый имеет abort point — частичное завершение даёт ценность. Sprint 1: kb-cli (Typer-based, новый entrypoint в `scripts/cli/`). Sprint 2: README rewrite + UI debug-pill hide. Sprint 3: LICENSE, install.sh, CI badge, ROADMAP, release tag.

**Tech Stack:**
- Backend: Python 3.12, FastAPI, Typer (new runtime dep), pytest, sqlite3
- Shell: bash (install.sh для Linux/macOS; не Windows)
- Frontend: vanilla JS (URLSearchParams для debug-mode toggle)
- CI: existing `.github/workflows/ci.yml` уже имеет `python-ci` job — добавляем badge, новый Typer-dep валидируется автоматически

**Spec:** [`docs/superpowers/specs/2026-05-25-mvp-completion-design.md`](../specs/2026-05-25-mvp-completion-design.md)

**Test conventions used in this repo:**
- `py -m pytest tests/<file>.py -v` — basic invocation (Windows `py` launcher; no venv per MEMORY.md)
- `tests/conftest.py` уже настроен с `STUBS_PATH` — новые тесты подхватывают окружение автоматически
- Commits через `@'...'@` single-quoted here-string (см. PowerShell-spec плана PDF citation)

---

## File Structure

**Files created:**
- `scripts/kb_cli.py` — Typer entrypoint dispatching subcommands
- `scripts/cli/__init__.py` — package init
- `scripts/cli/backup.py` — `kb-cli backup` implementation
- `scripts/cli/restore.py` — `kb-cli restore` implementation
- `scripts/cli/reindex.py` — `kb-cli reindex` implementation
- `scripts/cli/health.py` — `kb-cli health` implementation
- `LICENSE` — Apache-2.0 caconical text
- `CONTRIBUTING.md` — dev-setup + conventions
- `SECURITY.md` — vuln reporting + no-SLA disclosure
- `ROADMAP.md` — deferred Phase 2 visible
- `install.sh` — Linux/macOS one-click installer
- `docs/screenshots/chat-with-citations.png` — RAG-ответ с кликабельной цитатой
- `docs/screenshots/pdf-viewer-modal.png` — модал с подсвеченным фрагментом
- `docs/screenshots/upload-flow.png` — drag-and-drop в действии
- `docs/release_checklist.md` — manual smoke шаги перед `git tag v1.0.0`
- `docs/legacy_README.md` — архив текущего README (linked from new)
- `tests/test_kb_cli_backup.py`
- `tests/test_kb_cli_restore.py`
- `tests/test_kb_cli_reindex.py`
- `tests/test_kb_cli_health.py`
- `tests/test_install_sh_smoke.py`
- `tests/test_readme_outsider.py`
- `tests/test_ui_debug_hide.py`
- `tests/test_kb_compliance_mode_health.py`

**Files modified:**
- `pyproject.toml` — `[project.scripts]` для `kb-cli`
- `requirements-runtime.txt` — добавить `typer~=0.12`
- `app/api/kb_mvp.py` — `/health` extended (db_size, doc_count, chunk_count, disk_free, last_indexed_at, compliance_mode)
- `data/www/index.html` — `pill-row` hidden by default; visible if `?debug=1`
- `README.md` — full rewrite (outsider-first)
- `.env.example` — KB_COMPLIANCE_MODE placeholder + commentary
- `.gitignore` — `var/data/backups/`

---

## Sprint 1 — Internal-use solidity (~12ч)

**Goal:** kb-cli subcommands для backup/restore/reindex/health. После Sprint'а данные защищены, можно жить дальше.

**Abort point:** после Task 1.2 + 1.3 + 1.5 (~7ч) — backup/restore/health минимально готовы.

---

### Task 1.1: Add Typer dependency and scaffold `scripts/cli/` package

**Files:**
- Modify: `requirements-runtime.txt`
- Create: `scripts/cli/__init__.py`
- Create: `scripts/kb_cli.py`

- [ ] **Step 1: Inspect current requirements-runtime.txt**

Run:
```powershell
py -c "import typer" 2>&1
```
Expected: `ModuleNotFoundError: No module named 'typer'` (confirming Typer not installed).

If already installed (you used it before), it's fine — proceed.

- [ ] **Step 2: Append Typer to runtime requirements**

Open `requirements-runtime.txt`. Append at the end (after `uvicorn[standard]~=0.37`):

```
typer~=0.12
```

- [ ] **Step 3: Install Typer locally**

Run:
```powershell
py -m pip install "typer~=0.12"
```
Expected: successful install. Verify:
```powershell
py -c "import typer; print(typer.__version__)"
```
Expected: prints version like `0.12.x`.

- [ ] **Step 4: Create scripts/cli/ package init**

Create `scripts/cli/__init__.py`:

```python
"""kb-cli subcommands package.

Each subcommand lives in its own module (backup.py, restore.py,
reindex.py, health.py) and exposes a Typer ``app`` instance that
the top-level ``scripts/kb_cli.py`` mounts as a subcommand group.
"""
```

- [ ] **Step 5: Create the kb-cli entrypoint stub**

Create `scripts/kb_cli.py`:

```python
"""kb-cli entrypoint.

After ``pip install -e .`` (Sprint 1.6 wires the ``[project.scripts]``
entry), the ``kb-cli`` command is available on PATH and dispatches to
the subcommands in :mod:`scripts.cli`.

Until then, run with ``py -m scripts.kb_cli <subcommand>``.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="kb-cli",
    help="Operations CLI for KB.AI (backup, restore, reindex, health).",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _root_callback() -> None:
    """Root callback — kept thin; subcommands attach via decorators below."""


def main() -> None:
    """Console-script entry point referenced from pyproject.toml."""
    app()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Verify the stub runs**

Run:
```powershell
py -m scripts.kb_cli --help
```
Expected: prints Typer help with "Operations CLI for KB.AI..." and zero subcommands listed (subcommands are added in 1.2-1.5).

- [ ] **Step 7: Commit**

```powershell
git add requirements-runtime.txt scripts/cli/__init__.py scripts/kb_cli.py
git commit -m @'
feat(kb-cli): scaffold Typer-based entrypoint and cli/ package

Adds typer~=0.12 to runtime deps. Empty Typer app at scripts/kb_cli.py
that subcommands (backup/restore/reindex/health) will register against
in subsequent tasks. Package scripts/cli/ created with docstring init.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.2: `kb-cli backup` command

**Files:**
- Create: `scripts/cli/backup.py`
- Modify: `scripts/kb_cli.py` (register subcommand)
- Create: `tests/test_kb_cli_backup.py`

**Background:** Backup format = `tar.gz` containing `kb_mvp.sqlite`, `kb_files/*.pdf`, and `manifest.json`. Manifest schema specified in spec Section 4.2.3.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kb_cli_backup.py`:

```python
"""Tests for `kb-cli backup` command."""

from __future__ import annotations

import json
import sqlite3
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.cli.backup import backup_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    """Create a fake KB layout: var/data/kb_mvp.sqlite + var/data/kb_files/."""
    data_dir = tmp_path / "var" / "data"
    data_dir.mkdir(parents=True)

    # Create a non-empty SQLite to back up
    db_path = data_dir / "kb_mvp.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE meta(k TEXT, v TEXT)")
    conn.execute("INSERT INTO meta VALUES ('embedder', 'hash')")
    conn.commit()
    conn.close()

    # A blob file
    kb_files = data_dir / "kb_files"
    kb_files.mkdir()
    (kb_files / "1.pdf").write_bytes(b"%PDF-1.4\nhello\n")

    return tmp_path


def test_backup_creates_targz_with_expected_contents(runner: CliRunner, kb_dir: Path) -> None:
    """tar.gz contains kb_mvp.sqlite, kb_files/, manifest.json."""
    out_archive = kb_dir / "out.tar.gz"
    result = runner.invoke(
        backup_app,
        ["--data-dir", str(kb_dir / "var" / "data"), str(out_archive)],
    )
    assert result.exit_code == 0, result.output
    assert out_archive.exists()

    with tarfile.open(out_archive, "r:gz") as tf:
        names = {m.name for m in tf.getmembers()}
    assert "kb_mvp.sqlite" in names
    assert "kb_files/1.pdf" in names
    assert "manifest.json" in names


def test_backup_manifest_has_required_fields(runner: CliRunner, kb_dir: Path) -> None:
    out_archive = kb_dir / "out.tar.gz"
    result = runner.invoke(
        backup_app,
        ["--data-dir", str(kb_dir / "var" / "data"), str(out_archive)],
    )
    assert result.exit_code == 0, result.output

    with tarfile.open(out_archive, "r:gz") as tf:
        manifest_member = tf.extractfile("manifest.json")
        assert manifest_member is not None
        manifest = json.loads(manifest_member.read())

    assert manifest["version"] == "1.0"
    assert "created_at" in manifest
    assert manifest["kb_mvp_db_path"].endswith("kb_mvp.sqlite")
    assert manifest["file_count"] == 1
    assert manifest["total_bytes"] > 0
    assert manifest["embedder_used"] == "hash"


def test_backup_handles_empty_kb_files_dir(runner: CliRunner, tmp_path: Path) -> None:
    """If kb_files/ is empty or missing, manifest.file_count == 0."""
    data_dir = tmp_path / "var" / "data"
    data_dir.mkdir(parents=True)
    db_path = data_dir / "kb_mvp.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE meta(k TEXT, v TEXT)")
    conn.commit()
    conn.close()

    out = tmp_path / "out.tar.gz"
    result = runner.invoke(
        backup_app,
        ["--data-dir", str(data_dir), str(out)],
    )
    assert result.exit_code == 0, result.output

    with tarfile.open(out, "r:gz") as tf:
        manifest = json.loads(tf.extractfile("manifest.json").read())
    assert manifest["file_count"] == 0


def test_backup_fails_if_db_missing(runner: CliRunner, tmp_path: Path) -> None:
    """Missing kb_mvp.sqlite → non-zero exit with helpful message."""
    data_dir = tmp_path / "var" / "data"
    data_dir.mkdir(parents=True)

    out = tmp_path / "out.tar.gz"
    result = runner.invoke(
        backup_app,
        ["--data-dir", str(data_dir), str(out)],
    )
    assert result.exit_code != 0
    assert "kb_mvp.sqlite" in result.output.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_kb_cli_backup.py -v
```
Expected: FAIL with `ModuleNotFoundError: scripts.cli.backup`.

- [ ] **Step 3: Implement backup.py**

Create `scripts/cli/backup.py`:

```python
"""kb-cli backup — pack kb_mvp.sqlite + kb_files/ into a tar.gz with manifest."""

from __future__ import annotations

import io
import json
import sqlite3
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import typer

backup_app = typer.Typer(
    name="backup",
    help="Pack the KB SQLite database and original-file blobs into a tar.gz archive.",
    add_completion=False,
)


def _detect_embedder(db_path: Path) -> str:
    """Best-effort detection of the embedder name currently stored in kb_chunks.

    Returns 'unknown' if the column is missing (e.g. very early schema) or
    the table is empty.
    """
    if not db_path.is_file():
        return "unknown"
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks)")}
            if "embedder" in cols:
                row = conn.execute(
                    "SELECT embedder FROM kb_chunks ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    return str(row[0])
            # Some old test fixtures use a meta table — fallback
            cols_meta = {row[1] for row in conn.execute("PRAGMA table_info(meta)")}
            if {"k", "v"}.issubset(cols_meta):
                row = conn.execute("SELECT v FROM meta WHERE k = 'embedder'").fetchone()
                if row:
                    return str(row[0])
        finally:
            conn.close()
    except sqlite3.Error:
        pass
    return "unknown"


@backup_app.callback(invoke_without_command=True)
def backup(
    output: Path = typer.Argument(..., help="Path for the output .tar.gz archive."),
    data_dir: Path = typer.Option(
        Path("var/data"),
        "--data-dir",
        help="Root of KB data (contains kb_mvp.sqlite and kb_files/).",
        show_default=True,
    ),
) -> None:
    """Create a tar.gz archive of the KB state for backup or transfer."""

    db_path = data_dir / "kb_mvp.sqlite"
    if not db_path.is_file():
        typer.echo(
            f"ERROR: kb_mvp.sqlite not found at {db_path}. "
            "Pass --data-dir if your KB lives elsewhere.",
            err=True,
        )
        raise typer.Exit(code=2)

    kb_files = data_dir / "kb_files"
    file_paths = sorted(kb_files.glob("*.pdf")) if kb_files.is_dir() else []
    total_bytes = sum(p.stat().st_size for p in file_paths) + db_path.stat().st_size

    manifest = {
        "version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kb_mvp_db_path": str(db_path),
        "file_count": len(file_paths),
        "total_bytes": total_bytes,
        "embedder_used": _detect_embedder(db_path),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tf:
        tf.add(db_path, arcname="kb_mvp.sqlite")
        for path in file_paths:
            tf.add(path, arcname=f"kb_files/{path.name}")
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        tf.addfile(info, io.BytesIO(manifest_bytes))

    typer.echo(f"OK: wrote {output} ({manifest['file_count']} files, {manifest['total_bytes']} bytes)")
```

- [ ] **Step 4: Register backup subcommand in kb_cli.py**

Open `scripts/kb_cli.py`. After the existing `app = typer.Typer(...)` block, before `def _root_callback`, add:

```python
from scripts.cli.backup import backup_app

app.add_typer(backup_app, name="backup")
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```powershell
py -m pytest tests/test_kb_cli_backup.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 6: Smoke-test the CLI end-to-end**

Run:
```powershell
py -m scripts.kb_cli backup --help
```
Expected: usage info with `--data-dir` option visible.

- [ ] **Step 7: Commit**

```powershell
git add scripts/cli/backup.py scripts/kb_cli.py tests/test_kb_cli_backup.py
git commit -m @'
feat(kb-cli): add backup subcommand

`kb-cli backup <out.tar.gz>` packs kb_mvp.sqlite + kb_files/*.pdf with
a manifest.json (version, created_at, kb_mvp_db_path, file_count,
total_bytes, embedder_used). Default --data-dir=var/data. Fails fast
with helpful message if kb_mvp.sqlite missing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.3: `kb-cli restore` command (replace + merge modes)

**Files:**
- Create: `scripts/cli/restore.py`
- Modify: `scripts/kb_cli.py` (register subcommand)
- Create: `tests/test_kb_cli_restore.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_kb_cli_restore.py`:

```python
"""Tests for `kb-cli restore` command."""

from __future__ import annotations

import json
import sqlite3
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.cli.backup import backup_app
from scripts.cli.restore import restore_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_kb(tmp_path: Path, *, embedder: str = "hash") -> Path:
    data_dir = tmp_path / "var" / "data"
    data_dir.mkdir(parents=True)
    db = data_dir / "kb_mvp.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE meta(k TEXT, v TEXT)")
    conn.execute("INSERT INTO meta VALUES ('embedder', ?)", (embedder,))
    conn.commit()
    conn.close()
    (data_dir / "kb_files").mkdir()
    (data_dir / "kb_files" / "1.pdf").write_bytes(b"%PDF-1.4\nA\n")
    return data_dir


def _make_archive(runner: CliRunner, src_data_dir: Path, archive: Path) -> None:
    result = runner.invoke(
        backup_app,
        ["--data-dir", str(src_data_dir), str(archive)],
    )
    assert result.exit_code == 0, result.output


def test_restore_into_empty_target(runner: CliRunner, tmp_path: Path) -> None:
    """Fresh target dir → file extracted, no prompt needed."""
    src = _make_kb(tmp_path / "src")
    archive = tmp_path / "out.tar.gz"
    _make_archive(runner, src, archive)

    target = tmp_path / "target" / "var" / "data"
    target.mkdir(parents=True)
    result = runner.invoke(
        restore_app,
        ["--data-dir", str(target), "--yes", str(archive)],
    )
    assert result.exit_code == 0, result.output
    assert (target / "kb_mvp.sqlite").is_file()
    assert (target / "kb_files" / "1.pdf").is_file()


def test_restore_replace_requires_yes_when_non_empty(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Non-empty target without --yes → aborts."""
    src = _make_kb(tmp_path / "src")
    archive = tmp_path / "out.tar.gz"
    _make_archive(runner, src, archive)

    target = _make_kb(tmp_path / "target")
    result = runner.invoke(
        restore_app,
        ["--data-dir", str(target), str(archive)],
        input="n\n",
    )
    assert result.exit_code != 0


def test_restore_replace_with_yes_creates_backup_of_blobs(
    runner: CliRunner, tmp_path: Path
) -> None:
    """`--yes --mode replace` against non-empty: existing kb_files moved to *.bak-*."""
    src = _make_kb(tmp_path / "src")
    archive = tmp_path / "out.tar.gz"
    _make_archive(runner, src, archive)

    target = _make_kb(tmp_path / "target")
    # Add a unique file in target's kb_files so we can detect the backup
    (target / "kb_files" / "old.pdf").write_bytes(b"old")

    result = runner.invoke(
        restore_app,
        ["--data-dir", str(target), "--yes", "--mode", "replace", str(archive)],
    )
    assert result.exit_code == 0, result.output

    # New files restored
    assert (target / "kb_files" / "1.pdf").is_file()
    # Old kb_files preserved as a sibling directory like kb_files.bak-YYYYMMDD-HHMMSS
    siblings = list(target.glob("kb_files.bak-*"))
    assert siblings, "expected a kb_files.bak-* backup directory"
    assert (siblings[0] / "old.pdf").is_file()


def test_restore_merge_keeps_existing_files(
    runner: CliRunner, tmp_path: Path
) -> None:
    """--mode merge does not overwrite existing files; only adds new ones."""
    src = _make_kb(tmp_path / "src")
    archive = tmp_path / "out.tar.gz"
    _make_archive(runner, src, archive)

    target = _make_kb(tmp_path / "target")
    (target / "kb_files" / "2.pdf").write_bytes(b"target-only")

    result = runner.invoke(
        restore_app,
        ["--data-dir", str(target), "--yes", "--mode", "merge", str(archive)],
    )
    assert result.exit_code == 0, result.output
    # Existing file preserved
    assert (target / "kb_files" / "2.pdf").read_bytes() == b"target-only"
    # Archive file added
    assert (target / "kb_files" / "1.pdf").is_file()


def test_restore_warns_on_embedder_mismatch(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Different embedder in source vs target → warning printed but proceed."""
    src = _make_kb(tmp_path / "src", embedder="ollama:nomic")
    archive = tmp_path / "out.tar.gz"
    _make_archive(runner, src, archive)

    target = _make_kb(tmp_path / "target", embedder="hash")
    result = runner.invoke(
        restore_app,
        ["--data-dir", str(target), "--yes", "--mode", "replace", str(archive)],
    )
    assert result.exit_code == 0
    assert "embedder" in result.output.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_kb_cli_restore.py -v
```
Expected: FAIL with `ModuleNotFoundError: scripts.cli.restore`.

- [ ] **Step 3: Implement restore.py**

Create `scripts/cli/restore.py`:

```python
"""kb-cli restore — extract a kb-cli backup archive into a data directory."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tarfile
from datetime import datetime
from enum import Enum
from pathlib import Path

import typer

restore_app = typer.Typer(
    name="restore",
    help="Restore a kb-cli backup archive into a data directory.",
    add_completion=False,
)


class RestoreMode(str, Enum):
    replace = "replace"
    merge = "merge"


def _detect_target_embedder(db_path: Path) -> str:
    if not db_path.is_file():
        return "unknown"
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks)")}
            if "embedder" in cols:
                row = conn.execute(
                    "SELECT embedder FROM kb_chunks ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    return str(row[0])
            cols_meta = {row[1] for row in conn.execute("PRAGMA table_info(meta)")}
            if {"k", "v"}.issubset(cols_meta):
                row = conn.execute("SELECT v FROM meta WHERE k = 'embedder'").fetchone()
                if row:
                    return str(row[0])
        finally:
            conn.close()
    except sqlite3.Error:
        pass
    return "unknown"


def _is_non_empty(data_dir: Path) -> bool:
    return (data_dir / "kb_mvp.sqlite").is_file() or any(
        (data_dir / "kb_files").glob("*.pdf")
    ) if (data_dir / "kb_files").is_dir() else (data_dir / "kb_mvp.sqlite").is_file()


@restore_app.callback(invoke_without_command=True)
def restore(
    archive: Path = typer.Argument(..., help="Path to the .tar.gz produced by `kb-cli backup`."),
    data_dir: Path = typer.Option(
        Path("var/data"),
        "--data-dir",
        help="Target data directory (will receive kb_mvp.sqlite + kb_files/).",
    ),
    mode: RestoreMode = typer.Option(
        RestoreMode.replace,
        "--mode",
        help="replace: overwrite target; merge: add files but do not overwrite existing.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the interactive confirmation prompt."
    ),
) -> None:
    """Restore a backup archive into a data directory."""

    if not archive.is_file():
        typer.echo(f"ERROR: archive not found: {archive}", err=True)
        raise typer.Exit(code=2)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest from archive (without extracting yet)
    with tarfile.open(archive, "r:gz") as tf:
        manifest_member = tf.extractfile("manifest.json")
        if manifest_member is None:
            typer.echo("ERROR: archive missing manifest.json", err=True)
            raise typer.Exit(code=2)
        manifest = json.loads(manifest_member.read())

    if manifest.get("version") != "1.0":
        typer.echo(
            f"WARNING: unknown manifest version {manifest.get('version')!r} — "
            "proceeding anyway.",
            err=True,
        )

    src_embedder = manifest.get("embedder_used", "unknown")
    tgt_embedder = _detect_target_embedder(data_dir / "kb_mvp.sqlite")
    if src_embedder != "unknown" and tgt_embedder != "unknown" and src_embedder != tgt_embedder:
        typer.echo(
            f"WARNING: embedder mismatch — archive was created with {src_embedder!r}, "
            f"target has {tgt_embedder!r}. Search may need reindex after restore."
        )

    non_empty = _is_non_empty(data_dir)
    if non_empty and mode is RestoreMode.replace:
        if not yes:
            confirm = typer.confirm(
                f"Target {data_dir} is non-empty. Replace contents?",
                default=False,
            )
            if not confirm:
                typer.echo("Aborted by user.")
                raise typer.Exit(code=1)

        # Move existing kb_files/ to a timestamped sibling for safety
        existing_blobs = data_dir / "kb_files"
        if existing_blobs.is_dir() and any(existing_blobs.iterdir()):
            stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            sibling = data_dir / f"kb_files.bak-{stamp}"
            shutil.move(str(existing_blobs), str(sibling))
            typer.echo(f"Backed up existing kb_files/ → {sibling.name}")

    # Extract members. For merge mode, skip files that already exist.
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name == "manifest.json":
                continue
            target_path = data_dir / member.name
            if mode is RestoreMode.merge and target_path.exists():
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:  # directory entry
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.write_bytes(extracted.read())

    typer.echo(f"OK: restored {archive.name} into {data_dir} (mode={mode.value})")
```

- [ ] **Step 4: Register restore subcommand**

Open `scripts/kb_cli.py`. After the line `from scripts.cli.backup import backup_app`, add:

```python
from scripts.cli.restore import restore_app
```

After the `app.add_typer(backup_app, name="backup")` line, add:

```python
app.add_typer(restore_app, name="restore")
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```powershell
py -m pytest tests/test_kb_cli_restore.py -v
```
Expected: PASS (5 tests).

- [ ] **Step 6: Verify backup still passes (no regression)**

Run:
```powershell
py -m pytest tests/test_kb_cli_backup.py tests/test_kb_cli_restore.py -v
```
Expected: PASS (9 total).

- [ ] **Step 7: Commit**

```powershell
git add scripts/cli/restore.py scripts/kb_cli.py tests/test_kb_cli_restore.py
git commit -m @'
feat(kb-cli): add restore subcommand with replace and merge modes

`kb-cli restore <archive.tar.gz>` extracts kb_mvp.sqlite + kb_files/
into --data-dir. --mode=replace (default) overwrites and stashes any
pre-existing kb_files/ to kb_files.bak-<ts>/; --mode=merge keeps
existing files and adds new ones. Embedder-mismatch between archive
and target prints a warning but does not abort. --yes/-y skips the
confirmation prompt for scripted use.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.4: `kb-cli reindex` command

**Files:**
- Create: `scripts/cli/reindex.py`
- Modify: `scripts/kb_cli.py` (register subcommand)
- Create: `tests/test_kb_cli_reindex.py`

**Background:** Reindex creates `kb_chunks_new`, re-embeds each chunk via a configurable embedder, then atomically swaps. Supports `--from-document-id N` for resume. After successful swap, drops old `kb_chunks_new` table.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kb_cli_reindex.py`:

```python
"""Tests for `kb-cli reindex` command."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.cli.reindex import reindex_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def populated_db(tmp_path: Path):
    """Create a SQLite KB with 3 documents, 5 chunks via the real store."""
    from app.services.kb_store import KnowledgeBaseStore

    db_path = tmp_path / "kb_mvp.sqlite"
    store = KnowledgeBaseStore(db_path)
    store.add_document("doc1", text="alpha " * 100)
    store.add_document("doc2", text="beta " * 100)
    store.add_document("doc3", text="gamma " * 100)
    yield db_path


def _count_chunks(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()[0]
    finally:
        conn.close()


def _embedder_distribution(db: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT embedder, COUNT(*) FROM kb_chunks GROUP BY embedder"
        ).fetchall()
        return {row[0]: row[1] for row in rows}
    finally:
        conn.close()


def test_reindex_dry_run_does_not_modify_db(runner: CliRunner, populated_db: Path):
    """--dry-run prints plan without changing rows."""
    before_count = _count_chunks(populated_db)
    before_dist = _embedder_distribution(populated_db)

    result = runner.invoke(
        reindex_app,
        ["--db-path", str(populated_db), "--embedder", "hash", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output

    assert _count_chunks(populated_db) == before_count
    assert _embedder_distribution(populated_db) == before_dist


def test_reindex_swaps_embedder(runner: CliRunner, populated_db: Path):
    """After reindex, all chunks should report the new embedder name."""
    # The "hash" embedder is deterministic and dependency-free → safe to reindex onto itself.
    result = runner.invoke(
        reindex_app,
        ["--db-path", str(populated_db), "--embedder", "hash", "--force-yes"],
    )
    assert result.exit_code == 0, result.output

    dist = _embedder_distribution(populated_db)
    # All chunks should now be hash:<dim>
    assert all(e.startswith("hash") for e in dist.keys()), dist


def test_reindex_atomic_rollback_on_failure(
    runner: CliRunner, populated_db: Path, monkeypatch
):
    """If embedding fails midway, old kb_chunks survive unchanged."""
    before_count = _count_chunks(populated_db)

    def broken_embedder(*_args, **_kwargs):
        raise RuntimeError("synthetic embed failure")

    # Patch the embedder factory used by reindex
    monkeypatch.setattr(
        "scripts.cli.reindex._make_embedder",
        lambda name: broken_embedder,
    )

    result = runner.invoke(
        reindex_app,
        ["--db-path", str(populated_db), "--embedder", "hash", "--force-yes"],
    )
    assert result.exit_code != 0
    assert _count_chunks(populated_db) == before_count


def test_reindex_resume_skips_done_documents(
    runner: CliRunner, populated_db: Path
):
    """--from-document-id N processes only docs with id >= N."""
    result = runner.invoke(
        reindex_app,
        [
            "--db-path", str(populated_db),
            "--embedder", "hash",
            "--from-document-id", "2",
            "--force-yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "processed 2 document" in result.output.lower() or "doc 2" in result.output.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_kb_cli_reindex.py -v
```
Expected: FAIL with `ModuleNotFoundError: scripts.cli.reindex`.

- [ ] **Step 3: Implement reindex.py**

Create `scripts/cli/reindex.py`:

```python
"""kb-cli reindex — re-embed all chunks atomically using a new embedder."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

import typer

reindex_app = typer.Typer(
    name="reindex",
    help="Re-embed every chunk using the given embedder. Atomic (rollback on failure).",
    add_completion=False,
)


def _make_embedder(name: str) -> Callable[[str], tuple[bytes, str, int]]:
    """Return a function that embeds a single chunk text.

    For now we only support 'hash' (deterministic, no external dep). Other
    embedders can be wired by importing from app.services.kb_embeddings —
    deferred until needed.
    """
    import struct
    from app.services import kb_embeddings

    if name.startswith("hash"):
        embedder = kb_embeddings.HashingEmbedder()

        def _embed(text: str) -> tuple[bytes, str, int]:
            vec = embedder.embed(text)
            blob = struct.pack(f"{len(vec)}f", *vec)
            return blob, embedder.name, len(vec)

        return _embed
    raise typer.BadParameter(
        f"Unknown embedder {name!r}. Only 'hash' is wired in this MVP. "
        "Extend scripts/cli/reindex.py::_make_embedder for ollama/api backends."
    )


@reindex_app.callback(invoke_without_command=True)
def reindex(
    db_path: Path = typer.Option(
        Path("var/data/kb_mvp.sqlite"),
        "--db-path",
        help="Path to kb_mvp.sqlite.",
    ),
    embedder: str = typer.Option(..., "--embedder", help="Embedder name (e.g. 'hash')."),
    from_document_id: int = typer.Option(
        1, "--from-document-id", help="Resume from document_id >= N."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan without changes."),
    force_yes: bool = typer.Option(False, "--force-yes", help="Skip confirmation."),
) -> None:
    """Re-embed chunks atomically and swap the kb_chunks table."""

    if not db_path.is_file():
        typer.echo(f"ERROR: {db_path} does not exist", err=True)
        raise typer.Exit(code=2)

    conn = sqlite3.connect(str(db_path))
    try:
        chunk_rows = conn.execute(
            """
            SELECT id, document_id, chunk_index, text, page_number
            FROM kb_chunks
            WHERE document_id >= ?
            ORDER BY document_id, chunk_index
            """,
            (from_document_id,),
        ).fetchall()
    finally:
        conn.close()

    docs_seen = sorted({row[1] for row in chunk_rows})
    typer.echo(
        f"Plan: reindex {len(chunk_rows)} chunks across {len(docs_seen)} documents "
        f"with embedder={embedder!r}, starting from doc_id={from_document_id}"
    )

    if dry_run:
        typer.echo("DRY RUN — no changes made.")
        raise typer.Exit(code=0)

    if not force_yes:
        if not typer.confirm("Proceed?", default=False):
            typer.echo("Aborted by user.")
            raise typer.Exit(code=1)

    embed = _make_embedder(embedder)

    # Atomic strategy: write into kb_chunks_new, then BEGIN; DELETE; INSERT FROM ...; COMMIT.
    conn = sqlite3.connect(str(db_path))
    try:
        # Snapshot column list so we can mirror schema (page_number may or may not exist).
        chunk_cols = {row[1] for row in conn.execute("PRAGMA table_info(kb_chunks)")}
        has_page = "page_number" in chunk_cols

        conn.execute("DROP TABLE IF EXISTS kb_chunks_new")
        create_sql = (
            "CREATE TABLE kb_chunks_new ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "document_id INTEGER NOT NULL, "
            "chunk_index INTEGER NOT NULL, "
            "text TEXT NOT NULL, "
            "embedding BLOB NOT NULL, "
            "embedder TEXT NOT NULL, "
            "dim INTEGER NOT NULL"
            + (", page_number INTEGER" if has_page else "")
            + ")"
        )
        conn.execute(create_sql)
        conn.commit()

        processed_docs: set[int] = set()
        for row in chunk_rows:
            chunk_id, doc_id, chunk_idx, text, page_no = row
            blob, embedder_name, dim = embed(text)
            if has_page:
                conn.execute(
                    "INSERT INTO kb_chunks_new(document_id, chunk_index, text, embedding, embedder, dim, page_number) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (doc_id, chunk_idx, text, blob, embedder_name, dim, page_no),
                )
            else:
                conn.execute(
                    "INSERT INTO kb_chunks_new(document_id, chunk_index, text, embedding, embedder, dim) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, chunk_idx, text, blob, embedder_name, dim),
                )
            processed_docs.add(doc_id)

        # Atomic swap
        conn.execute("BEGIN")
        conn.execute("DELETE FROM kb_chunks WHERE document_id >= ?", (from_document_id,))
        cols = "document_id, chunk_index, text, embedding, embedder, dim" + (
            ", page_number" if has_page else ""
        )
        conn.execute(
            f"INSERT INTO kb_chunks({cols}) SELECT {cols} FROM kb_chunks_new"
        )
        conn.execute("DROP TABLE kb_chunks_new")
        conn.commit()
    except Exception:
        conn.rollback()
        try:
            conn.execute("DROP TABLE IF EXISTS kb_chunks_new")
            conn.commit()
        except sqlite3.Error:
            pass
        conn.close()
        raise
    finally:
        try:
            conn.close()
        except sqlite3.ProgrammingError:
            pass

    typer.echo(
        f"OK: processed {len(processed_docs)} document(s), "
        f"re-embedded {len(chunk_rows)} chunk(s)"
    )
```

- [ ] **Step 4: Register reindex subcommand**

Open `scripts/kb_cli.py`. Add import and registration alongside the others:

```python
from scripts.cli.reindex import reindex_app
# ...
app.add_typer(reindex_app, name="reindex")
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```powershell
py -m pytest tests/test_kb_cli_reindex.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 6: Run all CLI tests + existing kb_store tests for regression**

Run:
```powershell
py -m pytest tests/test_kb_cli_backup.py tests/test_kb_cli_restore.py tests/test_kb_cli_reindex.py tests/test_kb_store_pages.py tests/test_kb_mvp.py -v
```
Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add scripts/cli/reindex.py scripts/kb_cli.py tests/test_kb_cli_reindex.py
git commit -m @'
feat(kb-cli): add reindex subcommand with atomic table swap

`kb-cli reindex --embedder hash` re-embeds all chunks (or those with
document_id >= --from-document-id) into a kb_chunks_new staging table.
After successful re-embed, BEGIN; DELETE FROM kb_chunks; INSERT FROM
kb_chunks_new; DROP kb_chunks_new; COMMIT — atomic swap, rollback on
any error. --dry-run prints plan only. Only the hash embedder is wired
right now; ollama/api support is a one-function extension when needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.5: Extend `/api/kb/health` + `kb-cli health` command

**Files:**
- Modify: `app/api/kb_mvp.py` (extend `health` endpoint)
- Create: `scripts/cli/health.py`
- Modify: `scripts/kb_cli.py` (register subcommand)
- Create: `tests/test_kb_cli_health.py`
- Create: `tests/test_kb_compliance_mode_health.py`

- [ ] **Step 1: Write the failing test for extended /health**

Create `tests/test_kb_compliance_mode_health.py`:

```python
"""Tests for extended /api/kb/health fields."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.kb_mvp import router
from app.services.kb_store import KnowledgeBaseStore


@pytest.fixture
def app_with_store(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import config as _cfg
    if hasattr(_cfg, "get_settings"):
        _cfg.get_settings.cache_clear()

    store = KnowledgeBaseStore(tmp_path / "kb.sqlite")
    store.add_document("doc1", text="alpha " * 50)
    fastapi_app = FastAPI()
    fastapi_app.include_router(router, prefix="/api/kb")
    fastapi_app.state.kb_mvp_store = store
    return fastapi_app, tmp_path


def test_health_includes_kb_stats(app_with_store):
    app, _ = app_with_store
    client = TestClient(app)
    resp = client.get("/api/kb/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "kb_stats" in data
    stats = data["kb_stats"]
    assert stats["documents_count"] >= 1
    assert stats["chunks_count"] >= 1
    assert stats["db_size_bytes"] > 0
    assert "disk_free_bytes" in stats
    assert "last_indexed_at" in stats  # may be None on empty


def test_health_echoes_compliance_mode_when_unset(app_with_store, monkeypatch):
    monkeypatch.delenv("KB_COMPLIANCE_MODE", raising=False)
    app, _ = app_with_store
    client = TestClient(app)
    resp = client.get("/api/kb/health")
    data = resp.json()
    assert "compliance_mode" in data
    assert data["compliance_mode"] is None
    assert data["compliance_implemented"] is False


def test_health_echoes_compliance_mode_when_set(app_with_store, monkeypatch):
    monkeypatch.setenv("KB_COMPLIANCE_MODE", "ru_strict")
    app, _ = app_with_store
    client = TestClient(app)
    resp = client.get("/api/kb/health")
    data = resp.json()
    assert data["compliance_mode"] == "ru_strict"
    assert data["compliance_implemented"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_kb_compliance_mode_health.py -v
```
Expected: FAIL — `kb_stats` and `compliance_mode` not in response.

- [ ] **Step 3: Extend the /health endpoint**

Open `app/api/kb_mvp.py`. Find `def health` at line 598-608. Replace the entire function block with:

```python
@public.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Liveness probe with LLM, embedder, reranker, auth, KB stats and compliance."""

    import os as _os
    import shutil as _shutil
    import sqlite3 as _sqlite3
    from pathlib import Path as _Path

    store = _store_for(request)
    db_path = _Path(store.db_path)
    documents_count = 0
    chunks_count = 0
    db_size_bytes = 0
    last_indexed_at: Optional[str] = None
    if db_path.is_file():
        db_size_bytes = db_path.stat().st_size
        try:
            conn = _sqlite3.connect(str(db_path))
            try:
                row = conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()
                if row:
                    documents_count = int(row[0])
                row = conn.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()
                if row:
                    chunks_count = int(row[0])
                row = conn.execute(
                    "SELECT MAX(created_at) FROM kb_documents"
                ).fetchone()
                if row and row[0]:
                    last_indexed_at = str(row[0])
            finally:
                conn.close()
        except _sqlite3.Error:
            pass

    try:
        disk_free_bytes = _shutil.disk_usage(str(db_path.parent if db_path.parent.is_dir() else _Path.cwd())).free
    except OSError:
        disk_free_bytes = 0

    compliance_mode = _os.environ.get("KB_COMPLIANCE_MODE") or None

    return {
        "status": "ok",
        "llm": kb_llm.provider_status(),
        "embedder": kb_embeddings.embedder_status(),
        "reranker": kb_rerank.reranker_status(),
        "auth": auth_status(),
        "kb_stats": {
            "documents_count": documents_count,
            "chunks_count": chunks_count,
            "db_size_bytes": db_size_bytes,
            "disk_free_bytes": disk_free_bytes,
            "last_indexed_at": last_indexed_at,
        },
        "compliance_mode": compliance_mode,
        "compliance_implemented": False,
    }
```

Note: the function signature now takes `request: Request`. The `Request` type is already imported at the top of the file — verify with `grep -n "from fastapi import" app/api/kb_mvp.py`.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```powershell
py -m pytest tests/test_kb_compliance_mode_health.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Run existing kb_mvp tests for regression**

Run:
```powershell
py -m pytest tests/test_kb_mvp.py -v
```
Expected: PASS. If a test broke due to `kb_stats` etc. being added — it's a schema-assertion test; widen the assertion to use `.get()` or check subset.

- [ ] **Step 6: Write the failing test for kb-cli health**

Create `tests/test_kb_cli_health.py`:

```python
"""Tests for `kb-cli health` command."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from scripts.cli.health import health_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_health_ok_path(runner: CliRunner, httpx_mock=None, monkeypatch):
    """If /health returns 200 and status=ok → exit code 0."""
    captured: dict = {}

    class FakeResponse:
        status_code = 200
        def json(self):
            return {
                "status": "ok",
                "kb_stats": {
                    "documents_count": 3,
                    "chunks_count": 9,
                    "db_size_bytes": 12345,
                    "disk_free_bytes": 99999999,
                    "last_indexed_at": "2026-05-25T10:00:00Z",
                },
                "compliance_mode": None,
                "compliance_implemented": False,
                "llm": {"selected": "deepseek"},
            }

    def fake_get(url, **_):
        captured["url"] = url
        return FakeResponse()

    monkeypatch.setattr("scripts.cli.health.httpx.get", fake_get)
    result = runner.invoke(health_app, ["--base-url", "http://127.0.0.1:8000"])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "documents=3" in result.output


def test_health_warn_low_disk(runner: CliRunner, monkeypatch):
    """If disk_free_bytes < 100MB → WARN exit code 1."""
    class FakeResponse:
        status_code = 200
        def json(self):
            return {
                "status": "ok",
                "kb_stats": {
                    "documents_count": 1, "chunks_count": 1,
                    "db_size_bytes": 1, "disk_free_bytes": 50_000_000,
                    "last_indexed_at": None,
                },
                "compliance_mode": None, "compliance_implemented": False,
                "llm": {"selected": None},
            }

    monkeypatch.setattr("scripts.cli.health.httpx.get", lambda *a, **k: FakeResponse())
    result = runner.invoke(health_app, ["--base-url", "http://127.0.0.1:8000"])
    assert result.exit_code == 1
    assert "WARN" in result.output


def test_health_fail_on_http_error(runner: CliRunner, monkeypatch):
    """If /health returns non-2xx or connection fails → FAIL exit code 2."""
    def raises(*_, **__):
        raise ConnectionError("refused")
    monkeypatch.setattr("scripts.cli.health.httpx.get", raises)
    result = runner.invoke(health_app, ["--base-url", "http://127.0.0.1:8000"])
    assert result.exit_code == 2
    assert "FAIL" in result.output
```

- [ ] **Step 7: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_kb_cli_health.py -v
```
Expected: FAIL — `scripts.cli.health` not found.

- [ ] **Step 8: Implement health.py**

Create `scripts/cli/health.py`:

```python
"""kb-cli health — query /api/kb/health and format for humans + cron."""

from __future__ import annotations

import httpx
import typer

health_app = typer.Typer(
    name="health",
    help="Query /api/kb/health and report status with cron-friendly exit codes.",
    add_completion=False,
)

DISK_FREE_WARN_THRESHOLD = 100 * 1024 * 1024  # 100 MB


@health_app.callback(invoke_without_command=True)
def health(
    base_url: str = typer.Option(
        "http://127.0.0.1:8000",
        "--base-url",
        help="Base URL of the running kb_api server.",
    ),
    timeout: float = typer.Option(5.0, "--timeout", help="HTTP timeout in seconds."),
) -> None:
    """Health-check the running KB API. Exit 0=OK, 1=WARN, 2=FAIL."""

    url = f"{base_url.rstrip('/')}/api/kb/health"
    try:
        resp = httpx.get(url, timeout=timeout)
    except Exception as exc:
        typer.echo(f"FAIL: cannot reach {url}: {exc}", err=True)
        raise typer.Exit(code=2)

    if resp.status_code != 200:
        typer.echo(f"FAIL: HTTP {resp.status_code} from {url}", err=True)
        raise typer.Exit(code=2)

    try:
        data = resp.json()
    except ValueError:
        typer.echo(f"FAIL: response not JSON", err=True)
        raise typer.Exit(code=2)

    stats = data.get("kb_stats") or {}
    docs = stats.get("documents_count", 0)
    chunks = stats.get("chunks_count", 0)
    db_size = stats.get("db_size_bytes", 0)
    disk_free = stats.get("disk_free_bytes", 0)
    last_idx = stats.get("last_indexed_at") or "never"
    llm = (data.get("llm") or {}).get("selected") or "none"
    compliance = data.get("compliance_mode") or "off"

    summary = (
        f"documents={docs} chunks={chunks} db={db_size}B "
        f"disk_free={disk_free}B llm={llm} compliance={compliance} "
        f"last_indexed={last_idx}"
    )

    if data.get("status") != "ok":
        typer.echo(f"FAIL: {data.get('status')!r}: {summary}", err=True)
        raise typer.Exit(code=2)

    if disk_free and disk_free < DISK_FREE_WARN_THRESHOLD:
        typer.echo(f"WARN: low disk: {summary}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"OK: {summary}")
```

- [ ] **Step 9: Register health subcommand**

Open `scripts/kb_cli.py`. Add:

```python
from scripts.cli.health import health_app
# ...
app.add_typer(health_app, name="health")
```

- [ ] **Step 10: Run tests to verify they pass**

Run:
```powershell
py -m pytest tests/test_kb_cli_health.py tests/test_kb_compliance_mode_health.py -v
```
Expected: PASS (6 tests).

- [ ] **Step 11: Commit**

```powershell
git add app/api/kb_mvp.py scripts/cli/health.py scripts/kb_cli.py tests/test_kb_cli_health.py tests/test_kb_compliance_mode_health.py
git commit -m @'
feat(kb-cli, health): extend /health with kb_stats and compliance echo

/api/kb/health now reports documents_count, chunks_count, db_size_bytes,
disk_free_bytes, last_indexed_at and echoes KB_COMPLIANCE_MODE env
(compliance_implemented: false marks Phase 2 placeholder). `kb-cli
health --base-url ...` calls the endpoint, formats human-friendly,
and uses exit codes 0/1/2 for OK/WARN/FAIL so cron can alert on disk.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 1.6: Wire `kb-cli` as a console-script entry-point

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read current pyproject.toml**

Run:
```powershell
py -c "from pathlib import Path; print(Path('pyproject.toml').read_text())"
```
Confirm current contents — should match the version inspected earlier (no `[project.scripts]` section yet).

- [ ] **Step 2: Add [project.scripts] section**

Open `pyproject.toml`. After line 5 (`dynamic = ["dependencies", "optional-dependencies"]`), add:

```toml

[project.scripts]
kb-cli = "scripts.kb_cli:main"
```

The full `[project]` table now reads:

```toml
[project]
name = "kb-ai"
version = "0.1.0"
requires-python = ">=3.12"
dynamic = ["dependencies", "optional-dependencies"]

[project.scripts]
kb-cli = "scripts.kb_cli:main"
```

- [ ] **Step 3: Re-install the package**

Run:
```powershell
py -m pip install -e .
```
Expected: successful install. Verify the script is on PATH:

```powershell
kb-cli --help
```
Expected: prints help with all 4 subcommands (backup, restore, reindex, health).

If `kb-cli` is not found, your global Scripts directory may not be on PATH. Workaround for tests: `py -m scripts.kb_cli --help` always works.

- [ ] **Step 4: Commit**

```powershell
git add pyproject.toml
git commit -m @'
chore(packaging): expose kb-cli as a console-script

Adds [project.scripts] kb-cli = scripts.kb_cli:main to pyproject.toml.
After `pip install -e .`, the kb-cli command is available on PATH and
dispatches backup/restore/reindex/health subcommands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

- [ ] **Step 5: Final Sprint 1 verification**

Run:
```powershell
py -m pytest tests/test_kb_cli_backup.py tests/test_kb_cli_restore.py tests/test_kb_cli_reindex.py tests/test_kb_cli_health.py tests/test_kb_compliance_mode_health.py tests/test_kb_mvp.py tests/test_kb_store_pages.py -v
```
Expected: PASS for all new + existing tests.

Run lint:
```powershell
ruff check scripts/ app/api/kb_mvp.py
black --check scripts/ app/api/kb_mvp.py
```
Expected: PASS.

---

## Sprint 2 — Presentable state (~10ч)

**Goal:** UI debug-pill hide + README rewrite + screenshots. После Sprint'а — можно показать другу/коллеге.

**Abort point:** после Task 2.1 + 2.2 (~7ч) — UI чистый, README понятный.

---

### Task 2.1: UI debug-pill hide behind `?debug=1`

**Files:**
- Modify: `data/www/index.html` (pill-row in `<header>`, lines ~345-351)
- Create: `tests/test_ui_debug_hide.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ui_debug_hide.py`:

```python
"""Tests that the index.html debug-pill row is hidden by default."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "data" / "www" / "index.html"


def test_pill_row_has_data_debug_attribute():
    """The pill-row container must be marked with data-debug-pill='1'."""
    html = HTML.read_text(encoding="utf-8")
    assert re.search(
        r'<div\s+class="pill-row"[^>]*data-debug-pill="1"',
        html,
    ), "pill-row must have data-debug-pill='1' attribute"


def test_inline_js_hides_pill_row_when_debug_not_in_query():
    """The inline JS must check URLSearchParams and hide [data-debug-pill] by default."""
    html = HTML.read_text(encoding="utf-8")
    assert "URLSearchParams" in html
    assert "data-debug-pill" in html
    # Must reference 'debug' as the query param name
    assert "'debug'" in html or '"debug"' in html


def test_debug_query_param_keeps_pills_visible():
    """When ?debug=1 is present, the JS must NOT hide the pills (i.e. assigns display='')."""
    html = HTML.read_text(encoding="utf-8")
    # Look for the conditional: if !has('debug') → display='none'
    pattern = re.compile(
        r"URLSearchParams[\s\S]{0,200}?debug[\s\S]{0,200}?(none|hidden)",
        re.IGNORECASE,
    )
    assert pattern.search(html), "expected conditional hide on missing ?debug"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_ui_debug_hide.py -v
```
Expected: FAIL — `data-debug-pill` attribute not present.

- [ ] **Step 3: Add data-debug-pill attribute to pill-row**

Open `data/www/index.html`. Find line ~345:

```html
      <div class="pill-row">
```

Replace with:

```html
      <div class="pill-row" data-debug-pill="1">
```

- [ ] **Step 4: Add the inline debug-toggle script**

In `data/www/index.html`, find the very first `<script>` block in the body (around the existing inline JS — likely after `<script src="/js/kb-auth.js"></script>` if it exists; otherwise the first inline `<script>` tag in `<body>`). Add this **immediately after** the opening `<script>` tag (before any existing logic):

```javascript
    // Debug-pill toggle: hide infra pills unless ?debug=1 is in the URL.
    (function () {
      try {
        const params = new URLSearchParams(window.location.search);
        if (!params.has('debug')) {
          const pillRow = document.querySelector('[data-debug-pill]');
          if (pillRow) pillRow.style.display = 'none';
        }
      } catch (_) {
        /* localStorage / URL parsing failure — leave pills visible */
      }
    })();
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```powershell
py -m pytest tests/test_ui_debug_hide.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 6: Smoke-test in browser**

Run:
```powershell
py -m uvicorn scripts.dev_server_mvp:app --port 8001
```

Open `http://127.0.0.1:8001/` in a browser. Verify:
- The "health-pill, LLM…, emb…, rerank…, auth…" row is **NOT** visible.
- The "Ключ" button — still visible (it's outside the pill-row? Verify; if it's inside the pill-row, you may want to move it out so users can still toggle the key panel).

Then open `http://127.0.0.1:8001/?debug=1` — verify the pills are back.

If the "Ключ" button is inside the pill-row and now hidden, move it OUT in the HTML — it's user-facing, not debug:

```html
      <div class="pill-row" data-debug-pill="1">
        <span id="health-pill"...></span>
        ...
        <span id="auth-pill"...></span>
      </div>
      <button id="auth-toggle"...>Ключ</button>
```

Re-run the smoke check.

Stop the server with Ctrl-C.

- [ ] **Step 7: Commit**

```powershell
git add data/www/index.html tests/test_ui_debug_hide.py
git commit -m @'
feat(ui): hide infra debug pills behind ?debug=1 query param

The pill-row in <header> now has data-debug-pill="1" and an inline
script hides it by default. Open the UI with ?debug=1 in the URL to
restore the previous developer view (health/llm/embedder/rerank/auth
pills). The "Ключ" button is moved outside the pill-row so end-users
can still configure the API key without enabling debug mode.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 2.2: README rewrite + archive old

**Files:**
- Modify: `README.md` (full rewrite)
- Create: `docs/legacy_README.md` (archive of old content)
- Create: `tests/test_readme_outsider.py`

- [ ] **Step 1: Archive the current README**

Run:
```powershell
Copy-Item README.md docs/legacy_README.md
git add docs/legacy_README.md
git commit -m @'
docs: archive current README as docs/legacy_README.md

Preserves the existing 1300-line developer-oriented README for
reference before the upcoming rewrite to an outsider-first format.
The new README will link back here for advanced topics (LoRA training,
docker-compose ops, env-var reference).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_readme_outsider.py`:

```python
"""Structural checks on the rewritten README.md."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_starts_with_h1_and_tagline():
    """First lines should declare what + for whom — a clear hook."""
    lines = README.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# "), f"first line should be H1, got: {lines[0]!r}"


def test_readme_contains_quickstart_section():
    text = README.read_text(encoding="utf-8")
    assert re.search(r"^## .*Quickstart|## .*Быстрый старт", text, re.MULTILINE), (
        "expected an H2 like '## Quickstart' or '## Быстрый старт'"
    )


def test_readme_contains_what_inside_section():
    text = README.read_text(encoding="utf-8")
    assert re.search(r"^## .*Что внутри|## .*What.s inside", text, re.MULTILINE)


def test_readme_contains_not_for_you_section():
    text = README.read_text(encoding="utf-8")
    assert re.search(
        r"^## .*Не для вас|## .*Not for you", text, re.MULTILINE
    ), "expected an H2 like '## Не для вас если' (anti-positioning)"


def test_readme_references_screenshots():
    text = README.read_text(encoding="utf-8")
    for shot in (
        "chat-with-citations.png",
        "pdf-viewer-modal.png",
        "upload-flow.png",
    ):
        assert shot in text, f"README should reference {shot}"


def test_readme_links_to_legacy():
    text = README.read_text(encoding="utf-8")
    assert "docs/legacy_README.md" in text, (
        "README should link to docs/legacy_README.md for developer details"
    )


def test_readme_mentions_apache_license():
    text = README.read_text(encoding="utf-8")
    assert re.search(r"Apache.?2\.0", text), "license section should mention Apache-2.0"


def test_readme_has_ci_badge():
    text = README.read_text(encoding="utf-8")
    assert re.search(r"!\[CI\]\(.*workflows.*\)", text), (
        "expected a CI badge image link"
    )


def test_readme_quickstart_is_short():
    """The quickstart block (≤30 lines) keeps the 5-min promise honest."""
    text = README.read_text(encoding="utf-8")
    match = re.search(
        r"## .*(Quickstart|Быстрый старт)\n(.*?)(?=^## )", text,
        re.DOTALL | re.MULTILINE,
    )
    assert match, "Quickstart section missing"
    body = match.group(2)
    lines = [l for l in body.splitlines() if l.strip()]
    assert len(lines) <= 40, f"quickstart too long ({len(lines)} non-blank lines)"
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_readme_outsider.py -v
```
Expected: FAIL on most checks — current README doesn't follow this structure.

- [ ] **Step 4: Write the new README**

Overwrite `README.md` with:

```markdown
# KB.AI — корпоративная база знаний с нейропоиском

![CI](https://github.com/aiprocadm/baza_znaniy_ai/actions/workflows/ci.yml/badge.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Self-hosted AI-помощник по корпоративным документам. Загружаете PDF / DOCX / XLSX / PPTX —
получаете ответы с цитатами из ваших файлов и кликабельным переходом к нужной странице
с подсветкой фрагмента. Полностью на вашем сервере: ни одного запроса в зарубежные облака
не уходит, если вы того не настроите.

![Чат с цитатами](docs/screenshots/chat-with-citations.png)

## Быстрый старт

```bash
git clone https://github.com/aiprocadm/baza_znaniy_ai.git
cd baza_znaniy_ai
docker compose -f compose.yml up -d --build
# Откройте http://localhost/
# Загрузите PDF, дождитесь индексации, спросите.
```

Без Docker (Linux/macOS):

```bash
git clone https://github.com/aiprocadm/baza_znaniy_ai.git
cd baza_znaniy_ai
bash install.sh
# Откройте http://localhost:8000/
```

После запуска — загружайте документы во вкладке «Документы», задавайте вопросы во вкладке
«Вопрос-ответ». Цитаты в ответе кликабельны и открывают PDF в встроенном вьювере с
автоматической подсветкой фрагмента.

## Что внутри

- **RAG-пайплайн на FastAPI:** Docling для layout-aware парсинга → semantic chunking →
  hashing/Ollama/OpenAI-API эмбеддинги → SQLite-вектор-стор → опциональный
  cross-encoder reranker (BAAI/bge-reranker-v2-m3 для русского) → LLM с цитатами.
- **PDF citation viewer:** клик на цитату `[файл.pdf, стр. 12]` открывает модальное окно
  с PDF.js и подсветкой фрагмента через find API.
- **6 LLM-провайдеров из коробки:** DeepSeek, Groq, OpenRouter, OpenAI, Ollama, и любой
  OpenAI-совместимый custom endpoint. Auto-detect через `.env` без перезапуска.
- **Streaming SSE-ответы:** токены приходят на клиент по мере генерации; multi-turn
  диалоги с памятью в SQLite.
- **kb-cli ops:** `kb-cli backup / restore / reindex / health` для операций без UI.
- **i18n-ready UI:** все строки вынесены в `data/www/i18n/ru.json`; переключение на
  другие CIS-языки — добавление JSON-файла.

![PDF viewer](docs/screenshots/pdf-viewer-modal.png)

![Загрузка](docs/screenshots/upload-flow.png)

## Не для вас если

- У вас уже есть Notion AI / MS Copilot и вы довольны — мы не лучше для general-purpose.
- Вы хотите multi-tenant SaaS — KB.AI single-tenant. Для каждой команды/компании —
  отдельная инсталляция.
- У вас > 1М документов — SQLite-стор начнёт скрипеть. Используйте `app/api/v1/*`
  (legacy mature path с Qdrant) или дождитесь Phase 2 hybrid stack.
- Вам нужен SLA — это side-project под Apache-2.0, поддержка best-effort через GitHub Issues.

## Configuration

Базовый `.env`:

```env
# LLM (выберите один; auto-priority DeepSeek > Groq > OpenRouter > OpenAI)
DEEPSEEK_API_KEY=sk-...

# Опционально: реальный embedder (по умолчанию hashing fallback)
KB_EMBEDDINGS_BACKEND=ollama
OLLAMA_EMBED_MODEL=nomic-embed-text

# Опционально: API key для всех mutating endpoints
KB_API_KEY=$(openssl rand -hex 32)
```

Подробный список переменных, продвинутая конфигурация (LoRA, llama.cpp, Qdrant,
Postgres) — см. [`docs/legacy_README.md`](docs/legacy_README.md).

## Operations

`kb-cli` — операции без браузера:

```bash
kb-cli backup var/backups/$(date +%F).tar.gz   # бэкап KB
kb-cli restore var/backups/2026-05-01.tar.gz   # восстановление
kb-cli reindex --embedder hash                  # миграция эмбеддера
kb-cli health                                    # health-check для cron
```

Установить как entry-point — `pip install -e .` (см. install.sh).

## Архитектура

См. [`docs/architecture.md`](docs/architecture.md) — два HTTP-пути (`/api/kb/*` MVP
single-tenant и `/api/v1/*` mature multi-tenant), почему они параллельны, и когда
их объединять.

## Roadmap

См. [`ROADMAP.md`](ROADMAP.md) — что **НЕ** планируем (anti-roadmap из vision'а),
и при каких условиях это меняется.

## Contributing

См. [`CONTRIBUTING.md`](CONTRIBUTING.md). TL;DR: TDD, Conventional Commits,
`ruff + black + pytest` зелёные, маленькие PR.

## Security

См. [`SECURITY.md`](SECURITY.md). Reports → aiproc.adm@gmail.com. Нет bug bounty;
30-day grace period перед публикацией patch'а.

## License

Apache-2.0. См. [`LICENSE`](LICENSE).

## Legacy / advanced

Старый разработческий README (LoRA training, llama.cpp setup, full env-var reference,
Operations Console) — [`docs/legacy_README.md`](docs/legacy_README.md).
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```powershell
py -m pytest tests/test_readme_outsider.py -v
```
Expected: PASS (9 tests). Some tests reference files (LICENSE, install.sh, ROADMAP.md, CONTRIBUTING.md, SECURITY.md) that don't exist yet — they're referenced from README but the test only checks textual mention, so it should still pass.

If `test_readme_has_ci_badge` fails because the badge URL points to a workflow that doesn't exist — verify `.github/workflows/ci.yml` exists (it does per project exploration); the badge URL pattern is correct.

- [ ] **Step 6: Commit**

```powershell
git add README.md tests/test_readme_outsider.py
git commit -m @'
docs: rewrite README for outsider-first audience

The previous 1300-line README mixed quickstart with deep developer
reference. New README targets a first-time visitor: tagline in 3
lines, screenshots inline, "Не для вас если" anti-positioning,
short quickstart, links out to legacy_README.md for advanced topics.
CI badge and Apache-2.0 badge prominent. kb-cli section explains
new operational commands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 2.3: Screenshots in `docs/screenshots/`

**Files:**
- Create: `docs/screenshots/chat-with-citations.png`
- Create: `docs/screenshots/pdf-viewer-modal.png`
- Create: `docs/screenshots/upload-flow.png`

**Note:** This task requires running the app, taking real screenshots, and optimizing them. Cannot be done by code alone — needs human action via browser + screenshot tool. The plan below documents the procedure.

- [ ] **Step 1: Boot the MVP dev server**

Run in one PowerShell window:
```powershell
py -m uvicorn scripts.dev_server_mvp:app --port 8001
```

- [ ] **Step 2: Capture chat-with-citations.png**

1. Open `http://127.0.0.1:8001/` in Chrome/Firefox.
2. Upload a sample PDF (any short PDF with text — e.g. a sample regulation document).
3. Wait for indexing to finish.
4. Switch to «Вопрос-ответ» tab.
5. Ask a question whose answer appears in the PDF.
6. Wait for the response with citations to render.
7. Take a screenshot of the visible chat area with the cited buttons visible.
8. Save as `docs/screenshots/chat-with-citations.png`.
9. Optimize: `pngquant --quality 65-80 --output docs/screenshots/chat-with-citations.png --force docs/screenshots/chat-with-citations.png` (if pngquant not installed, skip).

Target file size: ≤500KB.

- [ ] **Step 3: Capture pdf-viewer-modal.png**

1. Click one of the citation buttons from Step 2.
2. The PDF.js modal opens with the page rendered and the snippet highlighted.
3. Take a screenshot of the modal.
4. Save as `docs/screenshots/pdf-viewer-modal.png`.
5. Optimize.

- [ ] **Step 4: Capture upload-flow.png**

1. Switch to «Документы» tab.
2. Drag a file from your Desktop onto the upload area (or take the screenshot mid-upload showing a progress indication).
3. Save as `docs/screenshots/upload-flow.png`.
4. Optimize.

- [ ] **Step 5: Verify file sizes and existence**

Run:
```powershell
Get-ChildItem docs/screenshots/ | Select-Object Name, Length
```
Expected: 3 PNG files, each ≤500KB.

- [ ] **Step 6: Stop the server**

Press Ctrl+C in the uvicorn window.

- [ ] **Step 7: Commit**

```powershell
git add docs/screenshots/
git commit -m @'
docs: add product screenshots for README

Three PNGs captured from local MVP run:
- chat-with-citations.png — RAG answer with clickable citation buttons
- pdf-viewer-modal.png — PDF.js modal showing highlighted snippet
- upload-flow.png — drag-and-drop file upload in action

Each ≤500KB via pngquant optimization.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

- [ ] **Step 8: Sprint 2 verification**

Run:
```powershell
py -m pytest tests/test_ui_debug_hide.py tests/test_readme_outsider.py -v
```
Expected: PASS (all tests, screenshots now referenced and existing).

---

## Sprint 3 — OSS-ready (~7.5ч)

**Goal:** LICENSE, install.sh, contribution docs, ROADMAP, release tag. После Sprint'а — можно публиковать.

**Abort point:** после Task 3.1 + 3.2 + 3.6 (~3ч) — минимум OSS-compliance.

---

### Task 3.1: LICENSE (Apache-2.0)

**Files:**
- Create: `LICENSE`

- [ ] **Step 1: Fetch the canonical Apache-2.0 license text**

Run (PowerShell):
```powershell
Invoke-WebRequest -OutFile LICENSE -Uri "https://www.apache.org/licenses/LICENSE-2.0.txt"
```

Verify:
```powershell
Get-Content LICENSE -TotalCount 5
```
Expected: starts with `Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/`.

- [ ] **Step 2: Verify file size and integrity**

Run:
```powershell
(Get-Item LICENSE).Length
```
Expected: ~11KB (between 11000 and 12000 bytes).

- [ ] **Step 3: Commit**

```powershell
git add LICENSE
git commit -m @'
docs: add Apache-2.0 LICENSE

Canonical text fetched from apache.org. Apache-2.0 is patent-grant-
inclusive, maximally compatible with vendored PDF.js (Apache-2.0)
and TinyLlama (Apache-2.0). Permits commercial use, modification,
and forking with attribution.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 3.2: CONTRIBUTING.md

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Write the file**

Create `CONTRIBUTING.md`:

```markdown
# Contributing to KB.AI

Thanks for considering a contribution. KB.AI is a side-project maintained
best-effort by one person, so please read the "What we will NOT accept"
section before opening a large PR.

## Dev setup (5 minutes)

```bash
git clone https://github.com/aiprocadm/baza_znaniy_ai.git
cd baza_znaniy_ai
python3.12 -m pip install -e .[dev]
pytest -q
```

For the MVP UI dev server (no Qdrant, no llama.cpp):
```bash
python -m uvicorn scripts.dev_server_mvp:app --reload --port 8001
```

## Conventions

- **Commit messages:** [Conventional Commits](https://www.conventionalcommits.org/) — `feat(area): ...`, `fix(area): ...`, `docs(area): ...`, `refactor(area): ...`, `chore(area): ...`.
- **Code style:** `ruff` + `black` (configured in `pyproject.toml`). Run `make format` before pushing.
- **Tests:** TDD when feasible. New features need at least one happy-path test and one edge-case test. `pytest -q` must pass.
- **PR size:** keep under ~400 added LoC. Larger refactors — open an Issue first to discuss scope.
- **Type hints:** required on new public APIs; encouraged everywhere.

## What we WILL accept

- Bug fixes with a regression test.
- New OpenAI-compatible LLM provider presets (add to `app/services/kb_llm.py:KNOWN_PRESETS`).
- New parsers for upload formats (extend `app/ingest/`).
- i18n translations (add `data/www/i18n/<lang>.json`).
- Documentation improvements.

## What we will NOT accept (without prior discussion)

- Multi-tenant / SaaS / billing features — explicitly out of scope (`ROADMAP.md`).
- Slack / Teams / Telegram bot integrations — fragmentation overhead.
- Mobile apps — responsive web covers it.
- Agentic / tool-use features — KB use-case is RAG, not autonomy.
- New abstractions ("framework" PRs without concrete user need).
- Forks of the embedder/reranker stack to add caching/parallelism without a benchmark showing >2x improvement on a real corpus.

If you're unsure, open an Issue first.

## Running the full suite

```bash
make lint   # ruff + black --check
make test   # pytest -q
```

CI runs both on every PR.
```

- [ ] **Step 2: Verify it renders correctly**

Run:
```powershell
Get-Content CONTRIBUTING.md -TotalCount 10
```
Expected: file exists, starts with `# Contributing to KB.AI`.

- [ ] **Step 3: Commit**

```powershell
git add CONTRIBUTING.md
git commit -m @'
docs: add CONTRIBUTING.md

Documents dev setup, conventions (Conventional Commits, ruff/black,
pytest), and an explicit "What we will NOT accept" section that
mirrors the vision document's anti-roadmap. Goal: prevent
well-meaning PRs that pull the project toward SaaS/agentic
features the author explicitly does not want.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 3.3: SECURITY.md

**Files:**
- Create: `SECURITY.md`

- [ ] **Step 1: Write the file**

Create `SECURITY.md`:

```markdown
# Security Policy

## Reporting a vulnerability

Email security reports to **aiproc.adm@gmail.com** with subject prefix `[KB.AI security]`.
Please include:

- Affected version (commit hash or release tag).
- Steps to reproduce.
- Impact assessment (data exposure, RCE, DoS, etc.).
- Suggested mitigation, if any.

You will receive an acknowledgement within **7 days**. A coordinated disclosure
timeline (typically 30 days) will be proposed.

## What we treat as security issues

- Authentication / authorization bypasses on protected endpoints.
- Path traversal, SSRF, command injection.
- SQL injection (we use parameterized queries; report any deviation).
- Sensitive data leakage (API keys, document contents) to unauthorized callers.
- CSRF on mutating endpoints.

## What we do NOT treat as security issues

- Lack of rate limiting on individual endpoints — by design, single-tenant.
- LLM hallucinations or factual errors in answers — known RAG limitation, mitigate at the prompt level.
- Lack of multi-tenant isolation — `/api/kb/*` is single-tenant by design.
- Reliance on `KB_API_KEY` for all mutations — single-shared-key model is intentional for MVP.

## Disclosure preferences

- **No bug bounty.** This is a side-project; we cannot pay.
- **Credit happily given** in release notes if requested.
- **30-day grace period** before public disclosure, extendable by mutual agreement.
- **No SLA** for response time beyond the 7-day acknowledgement target.

## Known limitations (not bugs)

- Single-tenant: one `KB_API_KEY` for the whole installation.
- No document-level RBAC: anyone with the key sees all documents.
- SQLite for the MVP store: not crash-tolerant under heavy concurrent writes.
- No audit-log retention policy: `audit_log` table grows unbounded — operator's responsibility to prune.

## Supported versions

Only the latest released tag receives security patches. Older tags are best-effort.
```

- [ ] **Step 2: Commit**

```powershell
git add SECURITY.md
git commit -m @'
docs: add SECURITY.md

Documents vulnerability reporting channel (aiproc.adm@gmail.com), the
scope of what we treat as security issues vs intentional design choices
(single-tenant, single key), and explicit "no SLA, no bug bounty" so
contributors do not over-expect.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 3.4: install.sh (Linux/macOS one-click)

**Files:**
- Create: `install.sh`
- Create: `tests/test_install_sh_smoke.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_install_sh_smoke.py`:

```python
"""Smoke tests for install.sh (Linux/macOS only)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "install.sh"


@pytest.mark.skipif(sys.platform == "win32", reason="install.sh is for Linux/macOS only")
def test_install_sh_exists_and_executable():
    assert INSTALL_SH.is_file()
    # Check shebang
    head = INSTALL_SH.read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("#!"), f"missing shebang, got: {head!r}"
    assert "bash" in head, f"shebang should reference bash: {head!r}"


@pytest.mark.skipif(sys.platform == "win32", reason="install.sh is for Linux/macOS only")
def test_install_sh_passes_shellcheck():
    """If shellcheck is on PATH, install.sh must be clean."""
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck not installed")
    result = subprocess.run(
        ["shellcheck", str(INSTALL_SH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"shellcheck: {result.stdout}\n{result.stderr}"


@pytest.mark.skipif(sys.platform == "win32", reason="install.sh is for Linux/macOS only")
def test_install_sh_dry_run_does_not_install():
    """--dry-run should print plan and exit 0 without modifying anything."""
    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"dry-run failed: {result.stderr}"
    assert "DRY RUN" in result.stdout or "would" in result.stdout.lower()


def test_install_sh_has_python_version_check():
    """Even on Windows, we can grep the file for the version check pattern."""
    content = INSTALL_SH.read_text(encoding="utf-8")
    assert "3.12" in content, "install.sh should require Python 3.12"
    assert "python3" in content, "install.sh should invoke python3"


def test_install_sh_copies_env_example():
    content = INSTALL_SH.read_text(encoding="utf-8")
    assert ".env.example" in content
    assert ".env" in content
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```powershell
py -m pytest tests/test_install_sh_smoke.py -v
```
Expected: FAIL — install.sh doesn't exist (or the platform-skipped tests are reported as skipped).

- [ ] **Step 3: Write install.sh**

Create `install.sh`:

```bash
#!/usr/bin/env bash
# install.sh — one-click installer for KB.AI on Linux/macOS.
#
# Usage:
#   bash install.sh                    # full install
#   bash install.sh --dry-run          # print what would happen, no changes
#
# Requirements:
#   - Python 3.12+
#   - pip
#   - Internet access for pip install
#
# After install, start the server with:
#   python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
# Or, for the MVP-only stack:
#   python -m uvicorn scripts.dev_server_mvp:app --port 8001

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

say() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] would: $*"
    else
        echo "[install] $*"
    fi
}

run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] $*"
    else
        "$@"
    fi
}

# 1. Python version check
say "Checking Python 3.12+"
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not on PATH. Install Python 3.12+ first." >&2
    exit 1
fi

PY_OK=$(python3 -c 'import sys; print("1" if sys.version_info >= (3, 12) else "0")')
if [[ "$PY_OK" != "1" ]]; then
    echo "ERROR: Python 3.12+ required (got $(python3 --version))." >&2
    exit 1
fi

# 2. Install runtime + MVP deps
say "Installing dependencies (this may take a few minutes)"
run python3 -m pip install --upgrade pip
run python3 -m pip install -e .

# 3. Copy .env.example → .env if not present
if [[ ! -f .env ]]; then
    say "Copying .env.example → .env"
    run cp .env.example .env
    echo "  Edit .env to add your LLM API keys (DEEPSEEK_API_KEY etc.)"
else
    say ".env already exists — leaving untouched"
fi

# 4. Create var/data directory
say "Ensuring var/data/ exists"
run mkdir -p var/data

# 5. Final message
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Install complete."
echo "  Start the server:"
echo "    python3 -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000"
echo "  Or MVP-only:"
echo "    python3 -m uvicorn scripts.dev_server_mvp:app --port 8001"
echo "  Then open http://localhost:8000/"
echo "════════════════════════════════════════════════════════════════"
```

- [ ] **Step 4: Make it executable (Linux/macOS)**

If on Linux/macOS, run:
```bash
chmod +x install.sh
```

If on Windows, this is a no-op. The Apache git client preserves the executable bit when committed.

- [ ] **Step 5: Run shellcheck (if available)**

If shellcheck is installed:
```powershell
shellcheck install.sh
```
Expected: zero warnings. If you have any, fix inline.

If shellcheck not installed locally, the existing `.github/workflows/ci.yml` job `shell-lint` will catch issues on PR.

- [ ] **Step 6: Run the tests**

Run:
```powershell
py -m pytest tests/test_install_sh_smoke.py -v
```
Expected: PASS for content checks; SKIPPED for Linux-only behaviour tests if on Windows.

- [ ] **Step 7: Commit**

```powershell
git add install.sh tests/test_install_sh_smoke.py
git commit -m @'
feat(install): add bash install.sh for Linux/macOS

One-click installer: checks Python 3.12+, runs `pip install -e .`,
copies .env.example -> .env (if absent), creates var/data/, prints
final start commands. --dry-run flag previews actions without changes.
Tests cover existence, shebang, version check pattern, dry-run, and
shellcheck cleanliness. Smoke tests skipped on Windows.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 3.5: Verify CI covers kb-cli + add CI badge

**Files:**
- Modify: `README.md` (CI badge already added in Task 2.2, just verify URL)

**Background:** Spec section 4.1 originally said `ci.yml ← NEW` but the file existed. The `python-ci` job runs `pytest -q` over the whole `tests/` directory, so `test_kb_cli_*.py` and `test_install_sh_smoke.py` are picked up automatically. The `shell-lint` job lints `install.sh` automatically. No workflow changes needed.

- [ ] **Step 1: Verify the badge URL in README is correct**

Run:
```powershell
Select-String "actions/workflows/ci.yml/badge.svg" README.md
```
Expected: one match in the badge block at the top of README.

The badge URL should be:
```
https://github.com/aiprocadm/baza_znaniy_ai/actions/workflows/ci.yml/badge.svg
```

If you used a different repo/branch in Task 2.2, update README.md now.

- [ ] **Step 2: Verify Typer install does not break existing CI flow**

The `python-ci` job runs:
```
python -m pip install -r requirements-runtime.txt -r requirements-llm.txt -r requirements-dev.txt
python -m pip install -e .
```

Since Sprint 1.1 added `typer~=0.12` to `requirements-runtime.txt`, the install step picks it up automatically. No CI YAML edit needed.

- [ ] **Step 3: (Optional) Push a branch to trigger CI locally**

Not strictly required for this plan, but if you want to confirm before the v1.0.0 tag:

```powershell
git push origin feat/kb-mvp-corporate-rag
```

Then watch the run on GitHub. Expected: `python-ci` and `shell-lint` jobs green; `legacy-compatibility-tests` and `docker-lint-build` may run depending on path filters — they should also stay green since this PR doesn't touch `backend/` or `Dockerfile`.

- [ ] **Step 4: No commit needed for this task** (badge added in Task 2.2; CI workflow already covers everything).

---

### Task 3.6: ROADMAP.md

**Files:**
- Create: `ROADMAP.md`

- [ ] **Step 1: Write the file**

Create `ROADMAP.md`:

```markdown
# Roadmap

**KB.AI is a side-project. It does not intend to become a startup.**

This document is **anti-roadmap-first**: it explicitly lists what we will NOT
build, because at 10-20 hours/week the most important decision is what to say
no to.

## Currently shipped (v1.0.0)

- MVP RAG pipeline: ingest → chunk → embed → search → answer with citations
- 6 LLM providers (DeepSeek, Groq, OpenRouter, OpenAI, Ollama, custom)
- 3 embedding backends (hashing, Ollama, OpenAI-compat API)
- Cross-encoder reranker (BGE multilingual)
- PDF citation viewer with text-search highlight (PDF.js)
- Multi-turn dialogues with SQLite history
- SSE streaming responses
- API key auth + DoS protection
- Audit log + admin endpoint
- i18n-ready UI (RU)
- kb-cli (backup/restore/reindex/health)
- One-click Linux/macOS install.sh

## Deferred (we will build this IF a real user asks)

These are valid feature requests but **none of them are guaranteed**.
We will revisit each only when a specific GitHub Issue describes the use case.

- **GigaChat / YandexGPT native integration.** Vision Phase 2; valuable for RU/CIS compliance customers. Deferred until: 1+ Issue from someone who can't use the current OpenAI-compat path.
- **LoRA Auto-Train UI.** Vision Phase 2; valuable for domain-specific customization. Deferred until: 1+ Issue from someone who wants to fine-tune on their own corpus and finds the existing `scripts/train_lora.py` too low-level.
- **Compliance Mode (per-country).** Env-flag scaffolding ships in v1.0; actual filtering of LLM providers deferred until first compliance-driven request. See `KB_COMPLIANCE_MODE` in `.env.example`.
- **Multi-tenant SaaS.** Deferred until 5+ paying customers exist for single-tenant version. Until then, run one installation per team.
- **Hybrid sparse+dense search.** Useful for large corpora; deferred until SQLite store starts showing query-time pain (>50ms p95 on a real corpus).
- **Document-level RBAC.** Single shared `KB_API_KEY` is the MVP choice. Deferred until a multi-user installation needs per-document permissions.

## Will NOT build (anti-roadmap)

These have been considered and rejected. Don't open an Issue for these without
a really specific use case.

- ❌ Slack / Teams / Telegram bot integrations — fragmentation overhead, can be done as a thin client by users.
- ❌ Mobile apps — responsive web covers 95% of usage.
- ❌ Real-time collaboration / multi-cursor editing.
- ❌ Agentic features / tool use / autonomous agents — wrong abstraction for KB use case.
- ❌ Workspaces / spaces / nested permissions.
- ❌ Vector-DB-as-a-service — Qdrant and Pinecone do this better.
- ❌ Cloud-hosted demo with persistent storage — operationally expensive, security-sensitive.
- ❌ "Migrate to Postgres for everything" — SQLite is the right scale for single-tenant; Postgres is opt-in for chat history only.

## Roadmap re-evaluation triggers

This document is re-read and (possibly) updated when:

- A new GitHub Issue requests a deferred item with concrete use case.
- 30 days have passed since v1.0.0 with 0 issues and 0 stars (project considered tilted toward "internal-only" outcome).
- A specific commercial inquiry arrives (`discovery-call` label).

## Contact

Feature requests → [GitHub Issues](https://github.com/aiprocadm/baza_znaniy_ai/issues). For discovery / commercial inquiries, tag the issue `discovery-call`.
```

- [ ] **Step 2: Commit**

```powershell
git add ROADMAP.md
git commit -m @'
docs: add ROADMAP.md with anti-roadmap-first framing

ROADMAP states upfront that KB.AI is a side-project, lists shipped
features, explicitly deferred items (with concrete trigger conditions),
and an anti-roadmap (will-not-build). Mirrors vision document's
discipline of declining well-meaning scope-creep PRs. Trigger conditions
make it easy to revisit decisions when a real user appears.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 3.7: .env.example update + KB_COMPLIANCE_MODE

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Inspect current .env.example**

Run:
```powershell
Get-Content .env.example -TotalCount 50
```

- [ ] **Step 2: Append the Compliance Mode section**

Open `.env.example`. Append at the end (after the last variable):

```env

# ──────────────────────────────────────────────────────────────────────
# Compliance Mode (Phase 2 заготовка — НЕ имплементирована в v1.0)
# ──────────────────────────────────────────────────────────────────────
# Когда задан, /api/kb/health эхо-возвращает значение в поле
# "compliance_mode". Реальная фильтрация LLM-провайдеров (запретить
# западные облака в ru_strict, и т.п.) — отложена до явного pilot-
# запроса. См. ROADMAP.md.
#
# Допустимые значения (для будущего использования):
#   ru_strict       — local llama.cpp + GigaChat + YandexGPT
#   kz_strict       — local + Ollama (RU облака запрещены)
#   by_strict       — аналогично KZ
#   cis_universal   — только local llama.cpp + Ollama
#
# KB_COMPLIANCE_MODE=
```

- [ ] **Step 3: Verify**

Run:
```powershell
Select-String "KB_COMPLIANCE_MODE" .env.example
```
Expected: at least one match.

- [ ] **Step 4: Commit**

```powershell
git add .env.example
git commit -m @'
docs(env): document KB_COMPLIANCE_MODE Phase 2 placeholder

Adds a documented section to .env.example describing the four planned
compliance modes (ru_strict, kz_strict, by_strict, cis_universal).
The env var is currently echoed back by /api/kb/health but does not
yet filter providers — that ships in Phase 2 only if a real customer
asks for it (see ROADMAP.md).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

---

### Task 3.8: release_checklist.md + .gitignore + tag v1.0.0

**Files:**
- Create: `docs/release_checklist.md`
- Modify: `.gitignore`

- [ ] **Step 1: Write the release checklist**

Create `docs/release_checklist.md`:

```markdown
# Release checklist — KB.AI v1.0.0

Run through this before `git tag` and `GitHub release`. Manual smoke
catches things automated tests can't.

## 1. Code state

- [ ] On branch `feat/kb-mvp-corporate-rag` (or main after merge).
- [ ] Working tree clean: `git status --short` shows nothing.
- [ ] All commits squash-friendly (or already squashed) for the release PR.

## 2. Tests + lint

- [ ] `py -m pytest -q` → all green
- [ ] `ruff check .` → clean
- [ ] `black --check .` → clean

## 3. Manual smoke (clean Ubuntu VM or container)

Spin up a clean Ubuntu 22.04+ environment.

- [ ] `git clone https://github.com/aiprocadm/baza_znaniy_ai.git`
- [ ] `cd baza_znaniy_ai && bash install.sh`
- [ ] Server starts: `python3 -m uvicorn scripts.dev_server_mvp:app --port 8001`
- [ ] Open `http://localhost:8001/` — UI loads, no debug pills visible.
- [ ] Open `http://localhost:8001/?debug=1` — debug pills visible.
- [ ] Upload a PDF (any short text PDF).
- [ ] Wait for indexing → document appears in list.
- [ ] Ask a question in «Вопрос-ответ» tab.
- [ ] Answer renders with citations.
- [ ] Click a citation → PDF.js modal opens, page rendered, snippet highlighted.
- [ ] Close modal, reload page — state preserved.

## 4. kb-cli smoke

- [ ] `kb-cli --help` — shows 4 subcommands.
- [ ] `kb-cli backup /tmp/backup.tar.gz` — succeeds, manifest valid.
- [ ] `kb-cli restore /tmp/backup.tar.gz --data-dir /tmp/restored --yes` — succeeds.
- [ ] `kb-cli health --base-url http://localhost:8001` — exit code 0, prints OK.

## 5. Docs

- [ ] README.md renders cleanly on GitHub (check after push).
- [ ] All 3 screenshots present in `docs/screenshots/` and visible in README.
- [ ] CI badge shows passing.
- [ ] LICENSE, CONTRIBUTING, SECURITY, ROADMAP all link from README.

## 6. Tag + release

- [ ] `git tag -a v1.0.0 -m "Release v1.0.0"`
- [ ] `git push origin v1.0.0`
- [ ] Create GitHub Release with changelog (see release-notes template below).
- [ ] Verify release page renders, ZIP downloads work.

## Release notes template

```markdown
# v1.0.0 — first stable release

KB.AI is now usable as a self-hosted RAG over corporate documents.

## Highlights

- PDF citation viewer with text-search highlight (PDF.js)
- 6 LLM providers + 3 embedding backends + cross-encoder reranker
- Multi-turn dialogues + SSE streaming
- kb-cli for backup/restore/reindex/health
- One-click Linux/macOS install.sh

## What's NOT in this release (deferred to Phase 2)

- GigaChat / YandexGPT native integration
- LoRA Auto-Train UI
- Compliance Mode actual implementation (env-flag scaffold only)
- Multi-tenant SaaS

See ROADMAP.md for the full anti-roadmap.

## Install

bash install.sh # Linux/macOS
docker compose up -d --build # Docker (any OS)
```
```

- [ ] **Step 2: Update .gitignore**

Open `.gitignore`. Find the line `var/*` and add (after line 28 which has `var/data/*`):

```
var/data/backups/
```

The relevant section now reads:

```
var/*
!var/.gitkeep
!var/data/
var/data/*
!var/data/.gitkeep
var/data/backups/
```

- [ ] **Step 3: Final test suite run**

Run:
```powershell
py -m pytest -q
```
Expected: all green (existing 100+ tests + new Sprint 1-3 tests).

Run lint:
```powershell
ruff check .
black --check .
```
Expected: clean. If black flags new files, run `black scripts/ tests/test_kb_cli_*.py tests/test_*.py` to fix.

- [ ] **Step 4: Commit the checklist + gitignore**

```powershell
git add docs/release_checklist.md .gitignore
git commit -m @'
docs(release): add manual smoke checklist + ignore var/data/backups/

release_checklist.md walks through pre-tag verification: clean tree,
green tests, clean Ubuntu VM smoke (install, upload PDF, ask, click
citation), kb-cli smoke, and docs render check. Includes a release-
notes template referencing ROADMAP and v1.0 highlights.

.gitignore now excludes var/data/backups/ so kb-cli backup outputs
do not pollute git status.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
'@
```

- [ ] **Step 5: Run the release checklist manually**

Open `docs/release_checklist.md` and walk through every checkbox. Tick them in the file as you go. If any step fails — STOP, fix the underlying issue, re-run.

- [ ] **Step 6: Tag and push v1.0.0**

After the checklist is fully green:

```powershell
git tag -a v1.0.0 -m "Release v1.0.0 — first stable MVP"
git push origin feat/kb-mvp-corporate-rag
git push origin v1.0.0
```

- [ ] **Step 7: Create GitHub Release**

Via `gh` CLI:

```powershell
gh release create v1.0.0 --title "v1.0.0 — first stable release" --notes-file - <<'EOF'
# v1.0.0 — first stable release

KB.AI is now usable as a self-hosted RAG over corporate documents.

## Highlights

- PDF citation viewer with text-search highlight (PDF.js)
- 6 LLM providers + 3 embedding backends + cross-encoder reranker
- Multi-turn dialogues + SSE streaming
- kb-cli for backup/restore/reindex/health
- One-click Linux/macOS install.sh

## What's NOT in this release (deferred to Phase 2)

- GigaChat / YandexGPT native integration
- LoRA Auto-Train UI
- Compliance Mode actual implementation (env-flag scaffold only)
- Multi-tenant SaaS

See ROADMAP.md for the full anti-roadmap.

## Install

bash install.sh # Linux/macOS
docker compose up -d --build # Docker (any OS)
EOF
```

Or via the GitHub web UI: Releases → Draft a new release → choose tag `v1.0.0` → paste the notes from `docs/release_checklist.md` template section.

---

## Sprint 4 — Publication (~4ч, optional)

**Goal:** Хабр-пост + GitHub Issues labels. Этот sprint опционален; решение принимается **после** того как Sprint 3 завершён и tag создан.

---

### Task 4.1: Хабр / Reddit publication

**Files:** none (external action)

- [ ] **Step 1: Decide whether to publish**

Open `docs/superpowers/specs/2026-05-25-mvp-completion-design.md`. Re-read Section 3 (Commerce-readiness criterion). If after Sprint 3 you've changed your mind about wanting outsider attention, **skip this task entirely**.

- [ ] **Step 2: Draft the post (offline)**

Suggested title: «Self-hosted RAG для русских корпоративных документов с PDF citation viewer».

Outline:
1. Проблема (1-2 параграфа): ChatGPT не подходит для compliance-чувствительных корпус, Notion AI — облако, Onyx — не оптимизирован под русский.
2. Решение (1 параграф): self-hosted, Apache-2.0, Docling parsing, BGE reranker for RU, PDF citation viewer.
3. Демо (3 скриншота из `docs/screenshots/`).
4. Quickstart (5 строк bash).
5. Что НЕ умеет (раздел "Не для вас если" из README).
6. Что дальше (link to ROADMAP).
7. Ссылка на репозиторий + лицензия.

- [ ] **Step 3: Publish**

Post on:
- Хабр (Хабы: `Машинное обучение`, `Open source`, `Поисковые технологии`).
- Reddit r/LocalLLaMA, r/selfhosted, r/RAG.

- [ ] **Step 4: Add the link to the GitHub release notes**

After publishing, edit the v1.0.0 release notes to include "Featured on Хабр: <link>" if you want.

- [ ] **Step 5: No commit needed for this task** (external action).

---

### Task 4.2: GitHub Issues labels setup

**Files:** none (GitHub UI action)

- [ ] **Step 1: Open GitHub repo Issues → Labels**

`https://github.com/aiprocadm/baza_znaniy_ai/labels`

- [ ] **Step 2: Create the labels**

| Label | Color | Description |
|---|---|---|
| `feature-request` | `#a2eeef` | New feature proposal (review against ROADMAP.md) |
| `bug` | `#d73a4a` | Something that demonstrably broken |
| `discovery-call` | `#0075ca` | Someone interested in commercial use / pilot |
| `deferred` | `#cccccc` | Decided against now per ROADMAP; revisit triggers documented |
| `good-first-issue` | `#7057ff` | Simple-scope issue for new contributors |

Create via `gh` CLI alternative:

```powershell
gh label create feature-request --color a2eeef --description "New feature proposal"
gh label create discovery-call --color 0075ca --description "Commercial interest / pilot inquiry"
gh label create deferred --color cccccc --description "Decided against now per ROADMAP"
# 'bug' and 'good-first-issue' usually exist by default
```

- [ ] **Step 3: Save a search for discovery-call issues**

In the GitHub Issues page, filter by `label:discovery-call` and bookmark the URL. This is your single "is anyone asking?" signal.

- [ ] **Step 4: No commit needed** (GitHub-side configuration).

---

## Final verification

After all Sprints (or after Sprint 3 abort point):

- [ ] **Run full test suite**

```powershell
py -m pytest -q
```
Expected: PASS (all old + new tests).

- [ ] **Run lint**

```powershell
ruff check .
black --check .
```
Expected: clean.

- [ ] **Verify CI is green on GitHub**

After push, check the Actions tab. All jobs should be passing on the `feat/kb-mvp-corporate-rag` branch.

- [ ] **Verify v1.0.0 release page**

`https://github.com/aiprocadm/baza_znaniy_ai/releases/tag/v1.0.0` — should show release notes and downloadable source archive.

---

## Acceptance criteria summary

By the end of this plan (Sprint 1 + 2 + 3):

- ✅ `kb-cli backup / restore / reindex / health` все работают, имеют тесты, --help
- ✅ `/api/kb/health` отдаёт extended fields (`kb_stats`, `compliance_mode`, `compliance_implemented`)
- ✅ `pip install -e .` создаёт `kb-cli` в PATH
- ✅ UI debug-pills скрыты по умолчанию, видны через `?debug=1`
- ✅ README понятен outsider'у за 3-5 минут; 3 screenshots; CI badge; ссылки на legacy/CONTRIBUTING/SECURITY/ROADMAP
- ✅ LICENSE (Apache-2.0), CONTRIBUTING.md, SECURITY.md, ROADMAP.md существуют и связаны
- ✅ install.sh работает на Linux/macOS, тесты покрывают структуру и shellcheck
- ✅ `.env.example` содержит KB_COMPLIANCE_MODE с документацией
- ✅ `docs/release_checklist.md` — manual smoke перед tag'ом
- ✅ `.gitignore` исключает `var/data/backups/`
- ✅ Tag `v1.0.0` создан, GitHub Release опубликован
- ✅ Все 100+ существующих тестов остаются зелёными
- ✅ Lint (`ruff check .`, `black --check .`) — чистый

Sprint 4 (опциональный):
- ✅ Хабр / Reddit post published
- ✅ GitHub Issues labels (`feature-request`, `discovery-call`, `deferred`) созданы

Total commits expected: **~18-22** (one per task or sub-step). All atomic, revertible.
