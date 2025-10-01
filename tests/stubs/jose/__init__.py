"""Stub implementation of :mod:`jose` for unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


class JWTError(Exception):
    pass


@dataclass
class _Token:
    payload: Dict[str, Any]


class jwt:
    @staticmethod
    def encode(payload: Dict[str, Any], key: str, algorithm: str = "HS256") -> str:
        return f"token:{algorithm}:{key}:{payload!r}"

    @staticmethod
    def decode(token: str, key: str, algorithms: list[str] | tuple[str, ...]) -> Dict[str, Any]:
        if not token.startswith("token:"):
            raise JWTError("Invalid token")
        return {"token": token, "key": key, "algorithms": algorithms}


__all__ = ["JWTError", "jwt"]
