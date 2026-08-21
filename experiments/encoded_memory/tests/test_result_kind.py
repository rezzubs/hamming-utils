"""Tests for EncodedFaultInjection's mutually exclusive result kinds (simple vs detailed)."""

import torch

from .conftest import _make_experiment, _result


def test_simple_result_by_default():
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()
    assert _result(experiment)["kind"] == "simple"


def test_detailed_result_when_compare_bitwise():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    result = _result(experiment)
    assert result["kind"] == "detailed"
    assert len(result["results"]) == 1
    # A bit flip always changes the underlying bit pattern, so the faulty
    # parameter tensors must disagree with the golden ones somewhere.
    assert len(result["results"][0]["bitmask"]) > 0


def test_detailed_result_when_compare_bitwise_f16():
    # Regression test: `_populate_golden`/`run` must cast the dataset's
    # (always-float32) inputs to the model's dtype before calling forward,
    # otherwise this raises a dtype-mismatch RuntimeError.
    experiment = _make_experiment(compare_bitwise=True, dtype=torch.float16)
    experiment.run()

    result = _result(experiment)
    assert result["kind"] == "detailed"
    assert len(result["results"]) == 1
    assert len(result["results"][0]["bitmask"]) > 0


def test_multiple_runs_append_one_result_each():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()
    experiment.run()

    assert len(_result(experiment)["results"]) == 2
