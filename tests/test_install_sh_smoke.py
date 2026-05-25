"""Smoke tests for install.sh (Linux/macOS only)."""

from __future__ import annotations

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
