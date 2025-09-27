"""Security helpers for the FastAPI compatibility layer."""

from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "HTTPAuthorizationCredentials",
    "HTTPBearer",
    "OAuth2PasswordBearer",
    "OAuth2PasswordRequestForm",
]


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


class HTTPAuthorizationCredentials:
    """Minimal credential container matching the FastAPI interface."""

    def __init__(self, *, scheme: Optional[str] = None, credentials: Optional[str] = None) -> None:
        self.scheme = scheme
        self.credentials = credentials


class HTTPBearer:
    """Stub dependency that raises until a real request handler is available."""

    def __init__(self, *, auto_error: bool = True) -> None:
        self.auto_error = auto_error

    async def __call__(self, *args: Any, **kwargs: Any) -> Optional[HTTPAuthorizationCredentials]:  # pragma: no cover - not used in tests
        if self.auto_error:
            raise RuntimeError("HTTPBearer stub does not handle requests")
        return None
