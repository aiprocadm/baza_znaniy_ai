"""Stub implementation of :class:`passlib.context.CryptContext`."""

from __future__ import annotations

from typing import Any


class CryptContext:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def hash(self, password: str) -> str:
        return f"hashed:{password}"

    def verify(self, password: str, hash_value: str) -> bool:
        return hash_value == f"hashed:{password}"
