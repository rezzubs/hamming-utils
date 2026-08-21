"""Tests for `encoded_memory.results.bit_position_histogram`."""

from encoded_memory.results import bit_position_histogram


def test_bit_position_histogram_decomposes_positions():
    # bit 0 set once, bit 1 set twice (once alone, once alongside bit 0)
    histogram = bit_position_histogram([0b01, 0b11, 0b10])
    assert histogram == {0: 2, 1: 2}


def test_bit_position_histogram_skip_multi_bit():
    histogram = bit_position_histogram([0b01, 0b11, 0b10], skip_multi_bit=True)
    assert histogram == {0: 1, 1: 1}


def test_bit_position_histogram_empty_for_no_bitmask():
    assert bit_position_histogram([]) == {}
