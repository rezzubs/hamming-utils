"""Matmul-level agreement tooling: run the same fault through every backend
on one weight shape and report how they compare.

Not a unit test. Equality is only expected for integer arithmetic, so this
reports similarity and time, not pass/fail; see the roadmap's oracle/workhorse
ladder and `docs/fault-lifting.md`. Doubles as a tolerance-based regression
check and is cheap enough to run per fault/shape, unlike a full model-level
campaign comparison.
"""

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor

from systolic._rust import Fault
from systolic.lifted_backend import LiftedBackend
from systolic.simulated_backend import SimulatedBackend
from systolic.torch_backend import TorchBackend


@dataclass(slots=True, frozen=True)
class BackendResult:
    """One backend's output and wall time for one matmul call."""

    name: str
    output: Tensor
    seconds: float


@dataclass(slots=True, frozen=True)
class Agreement:
    """How `other` compares against `baseline` (the oracle) on the same matmul."""

    baseline: str
    other: str
    max_abs_error: float
    max_relative_error: float
    top1_flip_fraction: float
    """Fraction of batch items whose argmax over the out_features axis
    differs between `baseline` and `other`."""


def _run(
    name: str,
    backend: SimulatedBackend | LiftedBackend,
    weights: Tensor,
    activations: Tensor,
    fault: Fault | None,
) -> BackendResult:
    backend.set_fault(fault)
    start = time.perf_counter()
    output = backend.matmul(weights, activations)
    seconds = time.perf_counter() - start
    return BackendResult(name=name, output=output, seconds=seconds)


def _compare(baseline: BackendResult, other: BackendResult) -> Agreement:
    diff = (baseline.output - other.output).abs()

    # A fault can saturate an output element to +-inf (or a bit pattern that's
    # literally NaN); when both backends compute the exact same inf/NaN there
    # they agree, but `inf - inf` and `nan - nan` both evaluate to NaN, which
    # would otherwise read as a divergence (and poison any later reduction,
    # since NaN propagates through `.max()`). Force those elements to zero
    # difference; a real divergence (e.g. one backend saturates and the other
    # doesn't) still produces a large or infinite value here, which is the
    # signal this comparison exists to catch.
    both_nan = (baseline.output == other.output) | (
        baseline.output.isnan() & other.output.isnan()
    )
    diff = diff.masked_fill(both_nan, 0.0)

    # `diff` is already zeroed at `both_nan` elements, but dividing by
    # `baseline.output`'s magnitude there can still be 0/inf or 0/NaN (NaN in
    # either case) - mask the ratio too rather than just the diff.
    denominator = baseline.output.abs().clamp_min(1e-30)
    relative_error = (diff / denominator).masked_fill(both_nan, 0.0)

    baseline_top1 = baseline.output.argmax(dim=0)
    other_top1 = other.output.argmax(dim=0)

    return Agreement(
        baseline=baseline.name,
        other=other.name,
        max_abs_error=diff.max().item(),
        max_relative_error=relative_error.max().item(),
        top1_flip_fraction=(baseline_top1 != other_top1).float().mean().item(),
    )


def compare_matmul(
    array: tuple[int, int],
    weights: Tensor,
    activations: Tensor,
    fault: Fault | None,
) -> tuple[list[BackendResult], list[Agreement]]:
    """Run one matmul through the fault-free golden path plus every systolic
    backend with `fault` applied, and report how each backend-under-test
    (`torch`, `lifted`) agrees with `simulated` (the oracle).
    """
    nrows, ncols = array

    results = [
        BackendResult(
            name="torch",
            output=TorchBackend().matmul(weights, activations),
            seconds=0.0,
        ),
        _run("simulated", SimulatedBackend(nrows, ncols), weights, activations, fault),
        _run("lifted", LiftedBackend(nrows, ncols), weights, activations, fault),
    ]

    oracle = next(result for result in results if result.name == "simulated")
    agreements = [
        _compare(oracle, result) for result in results if result is not oracle
    ]

    return results, agreements


@dataclass(slots=True, frozen=True)
class AgreementSummary:
    """One backend-under-test's aggregated agreement with the oracle.

    Computed across several sampled faults on one weight shape.
    """

    baseline: str
    other: str
    samples: int
    mean_max_abs_error: float
    max_max_abs_error: float
    mean_max_relative_error: float
    max_max_relative_error: float
    mean_top1_flip_fraction: float
    max_top1_flip_fraction: float
    mean_seconds: float


def summarize(
    samples: Sequence[tuple[list[BackendResult], list[Agreement]]],
) -> list[AgreementSummary]:
    """Aggregate `compare_matmul` results from many faults.

    Returns one `AgreementSummary` per backend-under-test.
    """
    by_other: dict[str, list[tuple[Agreement, float]]] = defaultdict(list)

    for results, agreements in samples:
        results_by_name = {result.name: result for result in results}
        for agreement in agreements:
            by_other[agreement.other].append(
                (agreement, results_by_name[agreement.other].seconds)
            )

    summaries: list[AgreementSummary] = []
    for other, pairs in by_other.items():
        abs_errors = [agreement.max_abs_error for agreement, _ in pairs]
        relative_errors = [agreement.max_relative_error for agreement, _ in pairs]
        flip_fractions = [agreement.top1_flip_fraction for agreement, _ in pairs]
        seconds = [seconds for _, seconds in pairs]

        summaries.append(
            AgreementSummary(
                baseline=pairs[0][0].baseline,
                other=other,
                samples=len(pairs),
                mean_max_abs_error=sum(abs_errors) / len(abs_errors),
                max_max_abs_error=max(abs_errors),
                mean_max_relative_error=sum(relative_errors) / len(relative_errors),
                max_max_relative_error=max(relative_errors),
                mean_top1_flip_fraction=sum(flip_fractions) / len(flip_fractions),
                max_top1_flip_fraction=max(flip_fractions),
                mean_seconds=sum(seconds) / len(seconds),
            )
        )

    return summaries
