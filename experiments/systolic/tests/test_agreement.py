import math

import torch

from systolic.agreement import Agreement, BackendResult, _compare, summarize

_UNUSED_OUTPUT = torch.zeros(1)
"""`BackendResult.output` isn't read by `summarize`, so a placeholder is fine."""


def _agreements(
    abs_error: float, relative_error: float, flip: float
) -> list[Agreement]:
    return [
        Agreement(
            baseline="simulated",
            other="lifted",
            max_abs_error=abs_error,
            max_relative_error=relative_error,
            top1_flip_fraction=flip,
        )
    ]


def test_summarize_aggregates_mean_and_max() -> None:
    samples = [
        _agreements(abs_error=1.0, relative_error=0.1, flip=0.0),
        _agreements(abs_error=3.0, relative_error=0.3, flip=0.5),
    ]

    [summary] = summarize(samples)

    assert summary.baseline == "simulated"
    assert summary.other == "lifted"
    assert summary.samples == 2
    assert summary.mean_max_abs_error == 2.0
    assert summary.max_max_abs_error == 3.0
    assert summary.mean_max_relative_error == 0.2
    assert summary.max_max_relative_error == 0.3
    assert summary.mean_top1_flip_fraction == 0.25
    assert summary.max_top1_flip_fraction == 0.5


def test_compare_treats_matching_infinities_as_agreement() -> None:
    baseline = BackendResult(name="simulated", output=torch.tensor([1.0, float("inf")]))
    other = BackendResult(name="lifted", output=torch.tensor([1.0, float("inf")]))

    agreement = _compare(baseline, other)

    assert agreement.max_abs_error == 0.0
    assert agreement.max_relative_error == 0.0


def test_compare_treats_matching_nan_as_agreement() -> None:
    baseline = BackendResult(name="simulated", output=torch.tensor([1.0, float("nan")]))
    other = BackendResult(name="lifted", output=torch.tensor([1.0, float("nan")]))

    agreement = _compare(baseline, other)

    assert agreement.max_abs_error == 0.0
    assert agreement.max_relative_error == 0.0


def test_compare_still_reports_real_divergence_at_infinity() -> None:
    """One backend saturating while the other doesn't is a real disagreement,
    not something to zero out - it must stay visible (as a large or infinite
    value), not be swallowed the way matching infinities/NaNs are."""
    baseline = BackendResult(name="simulated", output=torch.tensor([1.0, float("inf")]))
    other = BackendResult(name="lifted", output=torch.tensor([1.0, 2.0]))

    agreement = _compare(baseline, other)

    assert math.isinf(agreement.max_abs_error)
