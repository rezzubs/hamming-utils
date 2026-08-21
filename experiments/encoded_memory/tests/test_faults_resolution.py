"""Tests for `faults` resolution and validation."""

import pytest

from .conftest import _make_experiment, _result


def test_full_bit_error_rate_flips_every_bit():
    experiment = _make_experiment(compare_bitwise=True, faults=1.0)
    experiment.run()

    bitmask = _result(experiment)["results"][0]["bitmask"]
    # in_features=4, out_features=3 -> 4*3 weight + 3 bias = 15 float32
    # elements. A bit error rate of 1.0 flips every bit in the encoded
    # memory exactly once (Picker is a full permutation), so every
    # element's bit pattern ends up fully complemented, i.e. all 32 bits
    # set. Stored as the unsigned bit pattern rather than the signed int32
    # view's `-1`.
    assert len(bitmask) == 15
    assert all(value == (1 << 32) - 1 for value in bitmask)


def test_faults_greater_than_bit_count_raises():
    with pytest.raises(ValueError, match="faults"):
        _make_experiment(compare_bitwise=False, faults=100_000)


def test_bit_error_rate_greater_than_one_raises():
    with pytest.raises(ValueError, match="bit error rate"):
        _make_experiment(compare_bitwise=False, faults=1.5)
