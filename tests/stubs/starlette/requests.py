"""Minimal Starlette request object used by FastAPI compatibility helpers."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping


class Request:
    """Very small subset of :class:`starlette.requests.Request`."""

    def __init__(self, scope: Mapping[str, Any]) -> None:
        self.scope: dict[str, Any] = dict(scope)
        self._headers: MutableMapping[str, Any] = self.scope.setdefault("headers", {})

    @property
    def app(self) -> Any:  # pragma: no cover - trivial accessor
        return self.scope.get("app")

    @property
    def headers(self) -> MutableMapping[str, Any]:  # pragma: no cover - trivial
        return self._headers
