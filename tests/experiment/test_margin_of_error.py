"""Tests for Experiment.margin_of_error."""

from .conftest import make


def test_margin_of_error_no_results():
    assert make().margin_of_error() is None


def test_margin_of_error_one_result():
    assert make([1.0]).margin_of_error() is None


def test_margin_of_error_identical_values():
    assert make([3.0, 3.0]).margin_of_error() == 0.0
