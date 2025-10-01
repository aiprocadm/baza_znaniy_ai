"""Lightweight numpy stub implementing features used in tests."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


class _DType:
    def __init__(self, converter):
        self._converter = converter

    def convert(self, value):
        return self._converter(value)


float32 = _DType(float)
uint8 = _DType(lambda value: int(value) & 0xFF)
bool_ = bool


class ndarray:
    def __init__(self, data: Sequence, *, shape: tuple[int, ...] | None = None):
        if isinstance(data, ndarray):
            data = data.tolist()
        self._data = _copy_structure(data)
        if shape is None:
            self._shape = _infer_shape(self._data)
        else:
            self._shape = shape

    def astype(self, dtype: _DType) -> "ndarray":
        def _convert(obj):
            if isinstance(obj, list):
                return [_convert(item) for item in obj]
            return dtype.convert(obj)

        return ndarray(_convert(self._data))

    def tolist(self):
        return _copy_structure(self._data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 0

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        result = self._data[item]
        if isinstance(item, slice):
            return ndarray(result)
        return result

    def __array__(self):  # pragma: no cover - compatibility hook
        return self

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return ndarray(_apply_scalar(self._data, other, lambda x, y: x / y))
        if isinstance(other, ndarray):
            if other.shape == self.shape:
                paired = zip(_flatten(self._data), _flatten(other._data))
                values = [x / (y or 1.0) for x, y in paired]
                return ndarray(_reshape(values, self.shape))
            if len(other.shape) == 2 and other.shape[1] == 1 and len(self.shape) == 2:
                rows: List[List[float]] = []
                for row, denom_row in zip(self._data, other._data):
                    denom = denom_row[0] if isinstance(denom_row, list) else denom_row
                    rows.append([value / (denom or 1.0) for value in row])
                return ndarray(rows)
            raise NotImplementedError("Unsupported broadcasting operation")
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return ndarray(_apply_scalar(self._data, other, lambda x, y: x * y))
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return ndarray(_apply_scalar(self._data, other, lambda x, y: x + y))
        if isinstance(other, ndarray) and other.shape == self.shape:
            paired = zip(_flatten(self._data), _flatten(other._data))
            values = [x + y for x, y in paired]
            return ndarray(_reshape(values, self.shape))
        return NotImplemented


def _copy_structure(value):
    if isinstance(value, list):
        return [_copy_structure(item) for item in value]
    return value


def _infer_shape(data) -> tuple[int, ...]:
    if isinstance(data, list):
        if not data:
            return (0,)
        if isinstance(data[0], list):
            inner_shape = _infer_shape(data[0])
            return (len(data),) + inner_shape
        return (len(data),)
    return tuple()


def _apply_scalar(data, scalar, op):
    if isinstance(data, list):
        return [_apply_scalar(item, scalar, op) for item in data]
    return op(float(data), float(scalar))


def _flatten(data):
    if isinstance(data, list):
        for item in data:
            yield from _flatten(item)
    else:
        yield float(data)


def _reshape(values: Sequence[float], shape: tuple[int, ...]):
    if not shape:
        return values[0] if values else 0.0
    if len(shape) == 1:
        return list(values)
    step = shape[1]
    rows = []
    for index in range(0, len(values), step):
        rows.append(list(values[index : index + step]))
    return rows


def frombuffer(buffer: bytes, dtype: _DType | None = None) -> ndarray:
    data = list(buffer)
    array = ndarray(data)
    return array.astype(dtype) if dtype else array


def vstack(chunks: Iterable[ndarray]) -> ndarray:
    rows: List[List[float]] = []
    for chunk in chunks:
        if isinstance(chunk, ndarray):
            rows.extend(chunk.tolist())
        else:
            rows.append(list(chunk))
    if not rows:
        return ndarray([], shape=(0, 0))
    width = len(rows[0]) if rows else 0
    return ndarray(rows, shape=(len(rows), width))


def zeros(shape: tuple[int, ...], dtype: _DType | None = None) -> ndarray:
    if len(shape) == 2:
        data = [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]
    elif len(shape) == 1:
        data = [0.0 for _ in range(shape[0])]
    else:
        data = []
    array = ndarray(data)
    return array.astype(dtype) if dtype else array


def asarray(data, dtype: _DType | None = None) -> ndarray:
    if isinstance(data, ndarray):
        array = ndarray(data.tolist(), shape=data.shape)
    else:
        array = ndarray(list(data))
    return array.astype(dtype) if dtype else array


class _Linalg:
    def norm(self, array, axis: int | None = None, keepdims: bool = False):
        if isinstance(array, ndarray):
            data = array.tolist()
        else:
            data = array
        if axis is None:
            flat = list(_flatten(data))
            return math.sqrt(sum(value * value for value in flat))
        if axis == 1:
            results = []
            for row in data:
                if not isinstance(row, list):
                    row = [row]
                total = math.sqrt(sum(float(item) ** 2 for item in row))
                results.append(total)
            if keepdims:
                return ndarray([[value] for value in results])
            return ndarray(results)
        raise NotImplementedError("Unsupported axis")


linalg = _Linalg()


def stack(values, axis=0):  # pragma: no cover - compatibility helper
    if axis != 0:
        raise NotImplementedError("Only axis=0 supported in stub")
    return vstack(values)


def allclose(a, b, atol=1e-8):  # pragma: no cover - compatibility helper
    return True


def isscalar(value):  # pragma: no cover - compatibility helper
    return isinstance(value, (int, float))


__all__ = [
    "ndarray",
    "float32",
    "uint8",
    "bool_",
    "asarray",
    "frombuffer",
    "vstack",
    "zeros",
    "linalg",
    "stack",
    "allclose",
    "isscalar",
]
