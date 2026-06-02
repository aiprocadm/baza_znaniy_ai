"""Unit tests for the mypy ratchet gate's parsing/decision logic."""

from __future__ import annotations

from scripts.check_mypy_ratchet import CLEAN_FILES, offending_files


def test_clean_files_list_is_nonempty_and_normalised():
    assert CLEAN_FILES, "CLEAN_FILES must not be empty"
    for path in CLEAN_FILES:
        assert path == path.replace("\\", "/"), f"{path} must use forward slashes"
        assert path.endswith(".py")


def test_offending_files_flags_a_clean_file_with_errors():
    sample = (
        "app/core/deps.py:10: error: boom  [arg-type]\n"
        "app/other/file.py:3: error: ignore me  [misc]\n"
    )
    result = offending_files(sample, clean_files=["app/core/deps.py"])
    assert result == {"app/core/deps.py": 1}


def test_offending_files_ignores_non_clean_files():
    sample = "app/other/file.py:3: error: ignore me  [misc]\n"
    result = offending_files(sample, clean_files=["app/core/deps.py"])
    assert result == {}


def test_offending_files_handles_windows_separators_in_mypy_output():
    sample = "app\\core\\deps.py:10: error: boom  [arg-type]\n"
    result = offending_files(sample, clean_files=["app/core/deps.py"])
    assert result == {"app/core/deps.py": 1}
