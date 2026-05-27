"""Unit tests verifying behaviour of the lightweight pydantic stub."""

from __future__ import annotations

import math

import pytest

from tests.stubs import pydantic as pydantic_stub


def test_field_validator_modes_execute_in_order() -> None:
    """Ensure ``mode="before"`` and ``mode="after"`` validators run correctly."""

    calls: list[tuple[str, object]] = []

    class SampleModel(pydantic_stub.BaseModel):
        value: float = pydantic_stub.Field(...)

        @pydantic_stub.field_validator("value", mode="before")
        @classmethod
        def _before(cls, raw: object) -> object:
            calls.append(("before", raw))
            if isinstance(raw, str):
                return raw.strip()
            return raw

        @pydantic_stub.field_validator("value")
        @classmethod
        def _after(cls, converted: float) -> float:
            calls.append(("after", converted))
            return converted * 2

    instance = SampleModel(value="1.5")
    assert math.isclose(instance.value, 3.0)
    assert calls[0] == ("before", "1.5")
    assert calls[1] == ("after", 1.5)


def test_numeric_constraints_raise_validation_error() -> None:
    """Numeric metadata should reject invalid values during coercion."""

    class RangeModel(pydantic_stub.BaseModel):
        amount: float = pydantic_stub.Field(..., gt=0.0, le=2.0)

    with pytest.raises(pydantic_stub.ValidationError):
        RangeModel(amount=-1.0)

    with pytest.raises(pydantic_stub.ValidationError):
        RangeModel(amount=float("nan"))

    assert RangeModel(amount=1.5).amount == pytest.approx(1.5)


def test_after_validators_cannot_bypass_numeric_constraints() -> None:
    """Re-apply constraints after validators mutate the coerced value."""

    class ValidatedModel(pydantic_stub.BaseModel):
        amount: float = pydantic_stub.Field(..., gt=0.0)

        @pydantic_stub.field_validator("amount")
        @classmethod
        def _flip_sign(cls, value: float) -> float:
            return -abs(float(value))

    with pytest.raises(pydantic_stub.ValidationError):
        ValidatedModel(amount=1.0)


def test_lora_adapter_name_validation() -> None:
    """Adapter names must be non-empty after stripping whitespace."""

    from app.models.lora import LoraAdapterName
    from pydantic import ValidationError

    assert LoraAdapterName(name="demo").name == "demo"

    with pytest.raises(ValidationError):
        LoraAdapterName(name="  ")
