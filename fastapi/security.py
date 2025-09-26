"""Security helpers for the FastAPI compatibility layer."""

from __future__ import annotations

from typing import Any


class OAuth2PasswordBearer:
    """Lightweight stand-in for the OAuth2PasswordBearer dependency."""

    def __init__(self, tokenUrl: str, **_: Any) -> None:  # noqa: N803 - match FastAPI signature
        self.tokenUrl = tokenUrl

    async def __call__(self) -> str:  # pragma: no cover - not used in tests
        raise RuntimeError("OAuth2PasswordBearer stub does not handle requests")


class OAuth2PasswordRequestForm:
    """Simplified form that stores submitted credentials."""

    def __init__(self, *, username: str = "", password: str = "", **_: Any) -> None:
        self.username = username
        self.password = password
