"""Tests for the Picker bindings (faultforge._rust.Picker)."""

import hypothesis.strategies as st
import pytest
from faultforge._rust import Picker
from hypothesis import given, settings

_U64_MAX = 2**64 - 1
_seeds = st.integers(min_value=0, max_value=_U64_MAX)


@given(size=st.integers(min_value=0, max_value=1024), seed=_seeds)
@settings(max_examples=100)
def test_picker_yields_full_permutation(size: int, seed: int) -> None:
    picker = Picker(size, seed=seed)
    values = list(picker)
    assert sorted(values) == list(range(size))
    assert picker.size == 0


@given(
    data=st.integers(min_value=1, max_value=1024).flatmap(
        lambda size: st.tuples(st.just(size), st.integers(0, size))
    ),
    seed=_seeds,
)
@settings(max_examples=100)
def test_picker_from_returned(data: tuple[int, int], seed: int) -> None:
    size, num_to_pick = data

    picker = Picker(size, seed=seed)
    already_returned = {next(picker) for _ in range(num_to_pick)}

    remaining = set(Picker.from_returned(size, already_returned, seed=seed))

    assert remaining.isdisjoint(already_returned)
    assert remaining | already_returned == set(range(size))


def test_seed_is_deterministic() -> None:
    assert list(Picker(50, seed=123)) == list(Picker(50, seed=123))


def test_size_and_len_track_consumption() -> None:
    picker = Picker(3, seed=0)
    assert picker.initial_size == 3
    assert picker.size == 3
    assert len(picker) == 3

    next(picker)
    assert picker.size == 2
    assert len(picker) == 2


def test_reset_restores_size_and_yields_full_permutation() -> None:
    picker = Picker(5, seed=7)
    list(picker)
    assert picker.size == 0

    picker.reset()
    assert picker.size == 5
    assert sorted(picker) == list(range(5))


def test_exhausted_picker_raises_stop_iteration() -> None:
    picker = Picker(0, seed=0)
    with pytest.raises(StopIteration):
        next(picker)


def test_from_returned_rejects_out_of_range_value() -> None:
    with pytest.raises(ValueError):
        Picker.from_returned(5, {99})
