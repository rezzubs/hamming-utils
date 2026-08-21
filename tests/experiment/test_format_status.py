"""Tests for Experiment.format_status."""

from faultforge.experiment import AdditionalRuns, Stability

from .conftest import make


def test_format_status_no_results_is_none():
    assert make().format_status() is None


def test_format_status_shows_mean_once_two_scores():
    # Mean/margin-of-error display shows up as soon as there's enough data,
    # regardless of whether any stop condition is configured at all.
    exp = make([1.0, 2.0])
    status = exp.format_status()
    assert status is not None
    assert "mean" in status
    assert "±" in status


def test_format_status_omits_margin_with_one_score():
    exp = make([1.0])
    status = exp.format_status()
    assert status is not None
    assert "±" not in status


def test_format_status_omits_relative_moe_without_stability():
    exp = make([1.0, 2.0])
    status = exp.format_status([AdditionalRuns(5)])
    assert status is not None
    assert "Relative MoE" not in status


def test_format_status_shows_relative_moe_with_stability():
    exp = make([1.0, 2.0])
    status = exp.format_status([Stability(min_samples=0, threshold=1.0)])
    assert status is not None
    assert "Relative MoE" in status
