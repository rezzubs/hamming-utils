"""Matmul-level agreement tooling: run the same fault through every backend
on one weight shape and report how they compare.

Not a unit test. Equality is only expected for integer arithmetic, so this
reports similarity and time, not pass/fail; see the roadmap's oracle/workhorse
ladder and `docs/fault-lifting.md`. Doubles as a tolerance-based regression
check and is cheap enough to run per fault/shape, unlike a full model-level
campaign comparison.
"""

import time
from dataclasses import dataclass

from torch import Tensor

from faultforge._internal.systolic.lifted_backend import LiftedBackend
from faultforge._internal.systolic.simulated_backend import SimulatedBackend
from faultforge._internal.systolic.torch_backend import TorchBackend
from faultforge._rust.systolic import Fault


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
    denominator = baseline.output.abs().clamp_min(1e-30)

    baseline_top1 = baseline.output.argmax(dim=0)
    other_top1 = other.output.argmax(dim=0)

    return Agreement(
        baseline=baseline.name,
        other=other.name,
        max_abs_error=diff.max().item(),
        max_relative_error=(diff / denominator).max().item(),
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
