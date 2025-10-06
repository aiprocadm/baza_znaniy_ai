"""Selected Starlette datastructure stand-ins used within the tests."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableMapping
from tempfile import SpooledTemporaryFile
from typing import Any


def _normalise_key(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return str(value)


def _normalise_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return str(value)


class MutableHeaders(MutableMapping[str, str]):
    """Very small subset of :class:`starlette.datastructures.MutableHeaders`."""

    def __init__(
        self,
        headers: Iterable[tuple[str, str]] | MutableMapping[str, str] | None = None,
        *,
        raw: Iterable[tuple[str | bytes, str | bytes]] | None = None,
    ) -> None:
        self._items: list[tuple[str, str, str]] = []
        if raw is not None:
            for key, value in raw:
                self._append(_normalise_key(key), _normalise_value(value))
        elif headers is not None:
            if isinstance(headers, MutableMapping):
                iterator = headers.items()
            else:
                iterator = headers
            for key, value in iterator:
                self[key] = value

    def _append(self, original_key: str, value: str) -> None:
        self._items.append((original_key, original_key.lower(), value))

    def __getitem__(self, key: str) -> str:
        normalised = key.lower()
        for original, lowered, value in reversed(self._items):
            if lowered == normalised:
                return value
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        original_key = _normalise_key(key)
        lowered = original_key.lower()
        filtered = [item for item in self._items if item[1] != lowered]
        self._items = filtered
        self._append(original_key, _normalise_value(value))

    def __delitem__(self, key: str) -> None:
        lowered = key.lower()
        new_items = [item for item in self._items if item[1] != lowered]
        if len(new_items) == len(self._items):
            raise KeyError(key)
        self._items = new_items

    def __iter__(self) -> Iterator[str]:
        for original, _, _ in self._items:
            yield original

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def __contains__(self, key: object) -> bool:  # pragma: no cover - helper
        if not isinstance(key, str):
            return False
        lowered = key.lower()
        return any(item[1] == lowered for item in self._items)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        lowered = key.lower()
        removed_value: Any = None
        new_items: list[tuple[str, str, str]] = []
        found = False
        for item in self._items:
            if not found and item[1] == lowered:
                removed_value = item[2]
                found = True
                continue
            new_items.append(item)
        if not found:
            if default is not None:
                return default
            raise KeyError(key)
        self._items = new_items
        return removed_value

    def items(self) -> Iterator[tuple[str, str]]:  # pragma: no cover - convenience
        for original, _, value in self._items:
            yield original, value

    @property
    def raw(self) -> list[tuple[bytes, bytes]]:
        return [
            (original.encode("latin-1"), value.encode("latin-1"))
            for original, _, value in self._items
        ]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        items = ", ".join(f"{key!r}: {value!r}" for key, value in self.items())
        return f"MutableHeaders({{{items}}})"


class Headers(MutableHeaders):
    """Compatibility shim for :class:`starlette.datastructures.Headers`."""

    def __init__(
        self,
        headers: Iterable[tuple[str, str]] | MutableMapping[str, str] | None = None,
        *,
        raw: Iterable[tuple[str | bytes, str | bytes]] | None = None,
    ) -> None:
        super().__init__(headers=headers, raw=raw)


class UploadFile:
    """Simplified asynchronous interface mirroring Starlette's UploadFile."""

    def __init__(
        self,
        *,
        file: Any | None = None,
        filename: str | None = None,
        headers: Any | None = None,
        content_type: str | None = None,
    ) -> None:
        if file is None:
            file = SpooledTemporaryFile(mode="w+b")
        self.file = file
        self.filename = filename
        if isinstance(headers, MutableHeaders):
            self.headers = headers
        else:
            mapping: Iterable[tuple[str, str]] | MutableMapping[str, str] | None
            mapping = headers if headers is not None else None
            self.headers = MutableHeaders(mapping)
        if content_type is not None:
            self.content_type = content_type
            if content_type:
                self.headers["content-type"] = content_type
        else:
            self.content_type = self.headers.get("content-type")

    async def read(self, size: int = -1) -> bytes:
        data = self.file.read(size)
        if isinstance(data, str):
            return data.encode()
        if data is None:
            return b""
        return data

    async def write(self, data: bytes | str) -> None:  # pragma: no cover - helper
        if isinstance(data, str):
            data = data.encode()
        self.file.write(data)

    async def seek(self, offset: int, whence: int = 0) -> None:  # pragma: no cover
        self.file.seek(offset, whence)

    async def close(self) -> None:
        if hasattr(self.file, "close"):
            self.file.close()

    async def __aenter__(self) -> "UploadFile":  # pragma: no cover
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover
        await self.close()
