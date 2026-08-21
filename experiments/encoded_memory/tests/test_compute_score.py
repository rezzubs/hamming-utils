"""Tests for compute_score."""

import json

import pytest
from encoded_memory.experiment import ReliabilityMetric, compute_score

from .conftest import _make_experiment


@pytest.mark.parametrize(
    ("metric", "correct", "total", "expected"),
    [
        (ReliabilityMetric.Accuracy, 3, 4, 75.0),
        (ReliabilityMetric.AccuracyDegradation, 1, 4, 25.0),
        (ReliabilityMetric.Sdc, 3, 4, 25.0),
        (ReliabilityMetric.Top1Sdc, 1, 4, 75.0),
    ],
)
def test_compute_score_matches_each_metrics_formula(metric, correct, total, expected):
    assert compute_score(metric, correct, total) == expected


def test_compute_score_matches_live_experiment_score():
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()

    serialized = json.loads(experiment.serialize())
    correct = serialized["result"]["results"][0]
    total_items = serialized["total_items"]

    assert (
        compute_score(ReliabilityMetric.Accuracy, correct, total_items)
        == experiment.scores()[0]
    )
