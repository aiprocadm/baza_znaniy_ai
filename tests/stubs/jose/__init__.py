"""Stub implementation of :mod:`jose` for unit tests."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable


class JWTError(Exception):
    """Error raised when a token cannot be decoded or validated."""


@dataclass
class _Token:
    """Internal representation of a stub token."""

    algorithm: str
    key: str
    payload: Dict[str, Any]

    def dumps(self) -> str:
        """Serialize the token payload into a base64-encoded JSON string."""

        raw = json.dumps(
            {"alg": self.algorithm, "key": self.key, "payload": self.payload},
            separators=(",", ":"),
            sort_keys=True,
            default=_json_default,
        )
        return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")

    @classmethod
    def loads(cls, token: str) -> "_Token":
        """Deserialize a base64-encoded JSON string into a token object."""

        try:
            raw = base64.urlsafe_b64decode(token.encode("ascii"))
        except Exception as exc:  # pragma: no cover - defensive guard
            raise JWTError("Invalid token encoding") from exc

        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise JWTError("Invalid token payload") from exc

        try:
            algorithm = data["alg"]
            key = data["key"]
            payload = data["payload"]
        except KeyError as exc:
            raise JWTError("Malformed token data") from exc

        if not isinstance(payload, dict):
            raise JWTError("Malformed token payload")

        return cls(algorithm=algorithm, key=key, payload=payload)


def _json_default(value: Any) -> Any:
    """Best-effort serializer for non-JSON-native values."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _ensure_not_expired(payload: Dict[str, Any]) -> None:
    exp = payload.get("exp")
    if exp is None:
        return

    exp_dt = _normalize_exp(exp)
    now = datetime.now(timezone.utc)
    if exp_dt <= now:
        raise JWTError("Token has expired")


def _normalize_exp(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise JWTError("Invalid exp claim") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    raise JWTError("Unsupported exp claim type")


class jwt:
    @staticmethod
    def encode(payload: Dict[str, Any], key: str, algorithm: str = "HS256") -> str:
        token = _Token(algorithm=algorithm, key=key, payload=dict(payload))
        return f"token.{token.dumps()}"

    @staticmethod
    def decode(token: str, key: str, algorithms: Iterable[str] | tuple[str, ...]) -> Dict[str, Any]:
        if not token.startswith("token."):
            raise JWTError("Invalid token prefix")

        encoded = token.split(".", 1)[1]
        decoded_token = _Token.loads(encoded)

        # The stub keeps the requested key/algorithms available for callers that
        # inspect them, but it does not perform cryptographic validation.
        _ensure_not_expired(decoded_token.payload)

        return decoded_token.payload


__all__ = ["JWTError", "jwt"]
