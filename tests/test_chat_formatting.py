"""Unit tests for the `_format_answer` helper."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable, List


def _load_format_answer() -> Callable[[str, List[dict[str, Any]]], str]:
    module_path = Path("srv/projects/kb/app/main.py")
    source = module_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source)

    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_format_answer":
            function_module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(function_module)
            namespace: dict[str, Any] = {}
            exec("from typing import Any, List", namespace)
            compiled = compile(function_module, filename=str(module_path), mode="exec")
            exec(compiled, namespace)
            return namespace["_format_answer"]

    raise RuntimeError("_format_answer function not found in main.py")


_format_answer = _load_format_answer()


def test_format_answer_with_citations():
    answer = "  Ответ на вопрос.  "
    citations = [
        {"file": "doc1.pdf", "page": 3},
        {"file": "doc2.txt"},
    ]

    formatted = _format_answer(answer, citations)

    expected = "Ответ на вопрос.\n\n" "Источники:\n\n" "[1] doc1.pdf — страница 3\n" "[2] doc2.txt"

    assert formatted == expected


def test_format_answer_without_citations():
    answer = "  Ответ без источников.  "

    formatted = _format_answer(answer, citations=[])

    assert formatted == "Ответ без источников."
