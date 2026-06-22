"""Fail CI if any file proven mypy-clean has regressed to >0 errors.

This is a *ratchet*: it asserts only about files listed in CLEAN_FILES, so the
repo's overall mypy baseline can keep shrinking independently. Each drive-to-zero
pass appends the files it cleaned to CLEAN_FILES in the same PR.

Usage:  py -3 scripts/check_mypy_ratchet.py
Exit 0 if all CLEAN_FILES report 0 errors; exit 1 (with a report) otherwise.
"""

from __future__ import annotations

import subprocess
import sys

# Files proven to report 0 mypy errors. Append to this list as more are cleaned.
CLEAN_FILES: list[str] = [
    # PR #567 (deps + file_stats safe pass)
    "app/core/deps.py",
    "app/services/file_stats.py",
    # This pass (object/None-narrowing clean-set)
    "app/llm/lora_runtime.py",
    "app/services/kb_store.py",
    "app/api/v1/search.py",
    "app/services/vectorstore.py",
    "app/api/v1/users.py",
    # 2026-06-22 drive-to-zero pass
    "app/services/synthetic_qa.py",
    "app/worker/main.py",
    "app/llm/api_provider.py",
    "app/models/entities.py",
    "app/services/accounting.py",
    # 2026-06-23 drive-to-zero pass 2
    "app/api/kb_mvp/documents.py",
    "app/api/kb_mvp/__init__.py",
    "app/memory/__init__.py",
    "app/api/v1/admin.py",
    "app/models/engine_guard.py",
    "app/api/error_responses.py",
]


def offending_files(mypy_output: str, clean_files: list[str]) -> dict[str, int]:
    """Return {clean_file: error_count} for clean files with >0 errors."""
    wanted = set(clean_files)
    counts: dict[str, int] = {}
    for line in mypy_output.splitlines():
        if ": error:" not in line:
            continue
        path = line.split(":", 1)[0].replace("\\", "/")
        if path in wanted:
            counts[path] = counts.get(path, 0) + 1
    return counts


def run_mypy() -> str:
    """Run the repo's configured mypy over app/ and return combined output."""
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "app"],
        capture_output=True,
        text=True,
    )
    return proc.stdout + proc.stderr


def main() -> int:
    output = run_mypy()
    offenders = offending_files(output, CLEAN_FILES)
    if offenders:
        print("mypy ratchet FAILED — these clean files regressed:")
        for path, count in sorted(offenders.items()):
            print(f"  {path}: {count} error(s)")
        print("\nFix the new errors or, if intentional, remove the file from")
        print("CLEAN_FILES in scripts/check_mypy_ratchet.py (discouraged).")
        return 1
    print(f"mypy ratchet OK — all {len(CLEAN_FILES)} clean files report 0 errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
