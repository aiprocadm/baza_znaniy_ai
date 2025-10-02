"""Minimal psycopg.errors stub used in tests."""

from __future__ import annotations


class SyntaxError(Exception):
    """Stub equivalent of psycopg.errors.SyntaxError."""


__all__ = ["SyntaxError"]
