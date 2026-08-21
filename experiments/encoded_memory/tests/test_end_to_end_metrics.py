"""End-to-end coverage across reliability metrics and golden_is_encoded."""

import pytest
import torch
from encoded_memory import ReliabilityMetric

from .conftest import _make_experiment, _result


@pytest.mark.parametrize(
    "metric",
    [
        ReliabilityMetric.Accuracy,
        ReliabilityMetric.AccuracyDegradation,
        ReliabilityMetric.Sdc,
        ReliabilityMetric.Top1Sdc,
    ],
)
@pytest.mark.parametrize("golden_is_encoded", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_all_reliability_metrics_run_end_to_end(
    metric: ReliabilityMetric, golden_is_encoded: bool, dtype: torch.dtype
):
    experiment = _make_experiment(
        compare_bitwise=False,
        golden_is_encoded=golden_is_encoded,
        reliability_metric=metric,
        dtype=dtype,
    )
    experiment.run()

    scores = experiment.scores()
    assert len(scores) == 1
    if metric == ReliabilityMetric.AccuracyDegradation:
        # Signed: the faulty model can outscore golden on a tiny random
        # dataset, which shows up as a negative "degradation".
        assert -100.0 <= scores[0] <= 100.0
    else:
        assert 0.0 <= scores[0] <= 100.0


def test_golden_is_encoded_produces_bitmask_against_self_model():
    experiment = _make_experiment(compare_bitwise=True, golden_is_encoded=True)
    experiment.run()

    result = _result(experiment)
    assert result["kind"] == "detailed"
    # Bitwise comparison must decode `self._model` (there's no unencoded
    # golden to fall back on) and still find the injected bit flip.
    assert len(result["results"][0]["bitmask"]) > 0
