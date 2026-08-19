"""Property tests for `_corrupt`: forcing one bit of every float32 element
to a stuck value via a same-width unsigned-int view.
"""

import pytest
import torch
from faultforge._internal.systolic.lift_apply import _corrupt
from faultforge._rust.systolic import StuckAtKind
from hypothesis import given, settings
from hypothesis import strategies as st

_STUCK_AT = st.sampled_from([StuckAtKind.Zero, StuckAtKind.One])
_BIT_INDEX = st.integers(min_value=0, max_value=31)
# Includes NaN/inf/-0.0/subnormals: bit-level correctness must hold for the
# full float32 bit space, not just "ordinary" values.
_FLOAT32 = st.floats(width=32, allow_nan=True, allow_infinity=True)


@given(value=_FLOAT32, bit_index=_BIT_INDEX)
@settings(max_examples=200)
def test_corrupt_stuck_at_one_sets_bit(value: float, bit_index: int) -> None:
    tensor = torch.tensor([value], dtype=torch.float32)
    corrupted = _corrupt(tensor, StuckAtKind.One, bit_index)
    bits = int(corrupted.view(torch.uint32).item())
    assert (bits >> bit_index) & 1 == 1


@given(value=_FLOAT32, bit_index=_BIT_INDEX)
@settings(max_examples=200)
def test_corrupt_stuck_at_zero_clears_bit(value: float, bit_index: int) -> None:
    tensor = torch.tensor([value], dtype=torch.float32)
    corrupted = _corrupt(tensor, StuckAtKind.Zero, bit_index)
    bits = int(corrupted.view(torch.uint32).item())
    assert (bits >> bit_index) & 1 == 0


@given(value=_FLOAT32, stuck_at=_STUCK_AT, bit_index=_BIT_INDEX)
@settings(max_examples=200)
def test_corrupt_only_touches_target_bit(
    value: float, stuck_at: StuckAtKind, bit_index: int
) -> None:
    """Every bit other than `bit_index` must be unchanged from the input."""
    tensor = torch.tensor([value], dtype=torch.float32)
    original_bits = int(tensor.view(torch.uint32).item())

    corrupted = _corrupt(tensor, stuck_at, bit_index)
    corrupted_bits = int(corrupted.view(torch.uint32).item())

    other_bits_mask = 0xFFFFFFFF ^ (1 << bit_index)
    assert original_bits & other_bits_mask == corrupted_bits & other_bits_mask


def test_corrupt_bit_31_sign_bit_one_makes_negative() -> None:
    tensor = torch.tensor([1.5], dtype=torch.float32)
    corrupted = _corrupt(tensor, StuckAtKind.One, 31)
    assert corrupted.item() == -1.5


def test_corrupt_bit_31_sign_bit_zero_makes_positive() -> None:
    tensor = torch.tensor([-1.5], dtype=torch.float32)
    corrupted = _corrupt(tensor, StuckAtKind.Zero, 31)
    assert corrupted.item() == 1.5


def test_corrupt_rejects_non_float32() -> None:
    tensor = torch.tensor([1.5], dtype=torch.float64)
    with pytest.raises(ValueError):
        _corrupt(tensor, StuckAtKind.One, 0)


def test_corrupt_operates_elementwise() -> None:
    tensor = torch.tensor([1.5, -2.25, 0.0, float("inf")], dtype=torch.float32)
    corrupted = _corrupt(tensor, StuckAtKind.One, 31)
    expected = torch.tensor([-1.5, -2.25, -0.0, float("-inf")], dtype=torch.float32)
    torch.testing.assert_close(corrupted, expected, equal_nan=True)
