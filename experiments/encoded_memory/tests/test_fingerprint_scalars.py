"""Tests for fingerprint content recorded by EncodedFaultInjection."""

import json

import torch
from encoded_memory import (
    EncodedFaultInjection,
    ReliabilityMetric,
)

from .conftest import _make_experiment


def _fingerprint_scalars(experiment: EncodedFaultInjection) -> dict:
    return json.loads(experiment.serialize())["fingerprint"]["scalars"]


def test_fingerprint_records_resolved_faults_from_int_input():
    scalars = _fingerprint_scalars(_make_experiment(compare_bitwise=False, faults=3))
    assert scalars["faults"] == 3
    assert "bit_error_rate" not in scalars


def test_fingerprint_records_resolved_faults_from_bit_error_rate_input():
    # in_features=4, out_features=3 -> 4*3 weight + 3 bias = 15 float32
    # elements = 480 bits total, so a bit error rate of 0.5 resolves to 240.
    scalars = _fingerprint_scalars(_make_experiment(compare_bitwise=False, faults=0.5))
    assert scalars["faults"] == 240
    assert "bit_error_rate" not in scalars


def test_fingerprint_identical_for_equivalent_faults_and_bit_error_rate():
    # `faults=240` and `faults=0.5` resolve to the same count on this fixed
    # fake model (15 float32 elements = 480 bits), so they must produce
    # identical fingerprints - otherwise resuming a file recorded with one
    # input style using the other would spuriously fail the fingerprint
    # check.
    from_int = _fingerprint_scalars(_make_experiment(compare_bitwise=False, faults=240))
    from_rate = _fingerprint_scalars(
        _make_experiment(compare_bitwise=False, faults=0.5)
    )
    assert from_int == from_rate


def test_fingerprint_records_golden_and_compare_bitwise_and_metric():
    scalars = _fingerprint_scalars(
        _make_experiment(
            compare_bitwise=True,
            golden_is_encoded=True,
            reliability_metric=ReliabilityMetric.Sdc,
        )
    )
    assert scalars["golden"] == "encoded"
    assert scalars["compare_bitwise"] is True
    assert scalars["reliability_metric"] == "sdc"


def test_fingerprint_records_dtype():
    scalars = _fingerprint_scalars(_make_experiment(compare_bitwise=False))
    assert scalars["dtype"] == "f32"

    scalars = _fingerprint_scalars(
        _make_experiment(compare_bitwise=False, dtype=torch.float16)
    )
    assert scalars["dtype"] == "f16"


def test_fingerprint_omits_test_image_limit_by_default():
    scalars = _fingerprint_scalars(_make_experiment(compare_bitwise=False))
    assert "test_image_limit" not in scalars


def test_fingerprint_records_test_image_limit_from_batch_limit():
    scalars = _fingerprint_scalars(
        _make_experiment(compare_bitwise=False, dataset_batch_limit=1)
    )
    # batch_size=2 (fixed in `_make_experiment`) * dataset_batch_limit=1
    assert scalars["test_image_limit"] == 2
