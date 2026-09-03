"""Measuring how much each PE's profiled MAC inputs differ from the array as a whole.

The question this answers: can a later fault model treat every PE the same
(one distribution for the whole array), or does it need a separate
distribution per row, per column, or per PE? Pooling is much cheaper, so the
useful answer is "pooling is fine", and this module exists to look for
evidence against it.

Everything here is built from one comparison between two samples of numbers:

1. Sort each sample.
2. Read each as a step function - "what fraction of my values are at or
   below `x`".
3. Take the largest vertical distance between the two step functions.

That distance (`ecdf_gap`) is a single number between 0 and 1. It is 0 when
the two samples describe the same distribution and 1 when they don't overlap
at all. It ignores any monotone rescaling of the values, so plotting on a
linear or a logarithmic axis changes how the comparison *looks* without
changing the number at all.

The number on its own is not meaningful, because two samples drawn from the
*same* source still differ a little by chance. So every PE is also compared
against itself, by splitting its own samples into two halves. That
self-comparison is the noise floor: a PE only carries real structure if its
gap against the pooled reference is clearly larger than its gap against
itself.
"""

import enum
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from systolic.profiling import ProfilingArrays


class Regime(enum.StrEnum):
    """Which of the profiled input regimes to analyze.

    ZERO is absent on purpose: its inputs are structurally always
    `(0, 0, 0)`, so profiling records nothing for it.
    """

    First = "first"
    Active = "active"
    Drain = "drain"

    def variables_for(self) -> list[Variable]:
        """The variables that carry real information in `regime`.

        FIRST omits the partial sum and DRAIN omits activation and weight,
        because in those regimes they are structurally zero rather than
        observed.
        """
        match self:
            case Regime.First:
                return [Variable.Activation, Variable.Weight]
            case Regime.Active:
                return [Variable.Activation, Variable.Weight, Variable.PartialSum]
            case Regime.Drain:
                return [Variable.PartialSum]


class Variable(enum.StrEnum):
    """One of the three MAC inputs a PE sees."""

    Activation = "activation"
    Weight = "weight"
    PartialSum = "partial-sum"


_TRIPLE_AXIS = {
    Variable.Activation: 0,
    Variable.Weight: 1,
    Variable.PartialSum: 2,
}
"""Position of each variable along the trailing axis of a `*_triples` array."""


def variable_samples(
    arrays: ProfilingArrays, regime: Regime, variable: Variable
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uintp]]:
    """Pull one variable's per-PE samples out of a profiling artifact.

    Returns the samples, shaped `(array_rows, array_cols, capacity)`, and
    the matching per-PE fill counts, shaped `(array_rows, array_cols)`.
    Entries at or past a PE's fill count are padding, not observations.

    Raises:
        ValueError: If `variable` is structurally zero in `regime` and so
            was never recorded (see `variables_for`).
    """
    if variable not in regime.variables_for():
        available = ", ".join(other.value for other in regime.variables_for())
        raise ValueError(
            f"{variable.value!r} is always zero in the {regime.value} regime "
            f"and is not recorded. Available: {available}"
        )

    match regime:
        case Regime.First:
            samples = arrays.first_triples[:, :, :, _TRIPLE_AXIS[variable]]
            fill = arrays.first_fill
        case Regime.Active:
            samples = arrays.active_triples[:, :, :, _TRIPLE_AXIS[variable]]
            fill = arrays.active_fill
        case Regime.Drain:
            samples = arrays.drain_partial_sums
            fill = arrays.drain_fill

    return (samples, fill)


def ecdf_gap(left: npt.NDArray[np.floating], right: npt.NDArray[np.floating]) -> float:
    """How differently two samples are distributed, from 0 (identical) to 1.

    Reads each sample as a step function of "what fraction of my values are
    at or below `x`" and returns the largest vertical distance between the
    two.

    Unaffected by any monotone rescaling of the values, and by the order
    they arrive in.

    Raises:
        ValueError: If either sample is empty.
    """
    if left.size == 0 or right.size == 0:
        raise ValueError("cannot compare an empty sample")

    # Both step functions are flat except where a value was observed, so the
    # largest gap between them is always reached at one of those values.
    # Checking every observed value therefore finds the true maximum, with
    # no need to scan a grid of candidate thresholds.
    candidate_thresholds = np.concatenate([left, right])

    left_ecdf = _fraction_at_or_below(left, candidate_thresholds)
    right_ecdf = _fraction_at_or_below(right, candidate_thresholds)

    return float(np.abs(left_ecdf - right_ecdf).max())


def _fraction_at_or_below(
    sample: npt.NDArray[np.floating], thresholds: npt.NDArray[np.floating]
) -> npt.NDArray[np.float64]:
    """For each threshold, what fraction of `sample`'s values are `<=` it.

    Equivalent to `[(sample <= t).sum() / sample.size for t in thresholds]`,
    just computed with a sort plus a binary search instead of a rescan of
    `sample` per threshold: once `sample` is sorted, "how many values are
    `<=` t" and "where would t be inserted to keep it sorted" are the same
    number, and `np.searchsorted` finds that number by binary search rather
    than by counting one element at a time.
    """
    sorted_sample = np.sort(sample)
    count_at_or_below = np.searchsorted(sorted_sample, thresholds, side="right")
    return count_at_or_below / sample.size


@dataclass(slots=True, frozen=True)
class GapGrid:
    """Per-PE gaps against the pooled array, for one regime and variable.

    Both grids are shaped `(array_rows, array_cols)` and hold `NaN` for
    every PE excluded for lack of data, so they line up with the physical
    array and can be drawn directly as an image.
    """

    gaps: npt.NDArray[np.float64]
    """Each PE's gap against the pooled reference sample.

    The quantity of interest: how unlike the array as a whole this PE is.
    """

    self_gaps: npt.NDArray[np.float64]
    """Each PE's gap against a second, disjoint sample of its own inputs.

    The noise floor. A PE compared against itself should score near zero,
    so this is how large `gaps` can get from sampling luck alone.
    """

    sample_size: int
    """Number of values on each side of every comparison.

    Held equal everywhere because gaps shrink as samples grow: comparisons
    made at different sample sizes are not comparable to each other.
    """

    included: int
    """How many PEs had enough recorded samples to be compared."""

    total: int
    """How many PEs the array has in total."""

    def noise_floor(self) -> float:
        """The typical gap between two samples of the *same* PE.

        Treat any PE whose `gaps` entry is not clearly above this as
        indistinguishable from the pooled array.
        """
        return float(np.nanmedian(self.self_gaps))

    def largest_gap(self) -> float:
        """The most unlike-the-array any single PE is."""
        return float(np.nanmax(self.gaps))


def gap_grid(
    samples: npt.NDArray[np.float32],
    fill: npt.NDArray[np.uintp],
    *,
    sample_size: int,
    seed: int = 0,
) -> GapGrid:
    """Compare every PE's recorded values against the array as a whole.

    `samples` is `(array_rows, array_cols, capacity)` and `fill` says how
    many leading entries of each PE's row are real observations rather than
    padding; `variable_samples` produces both.

    Every PE needs `2 * sample_size` real observations to take part: one
    subsample is compared against the pooled reference and the other,
    disjoint one against the first, giving the noise floor. PEs with fewer
    are excluded and left as `NaN`.

    Raises:
        ValueError: If `sample_size` is not positive, or if fewer than two
            PEs have enough observations to compare.
    """
    if sample_size < 1:
        raise ValueError(f"sample_size must be positive, got {sample_size}")

    array_rows, array_cols, _capacity = samples.shape
    needed = 2 * sample_size
    has_enough = fill >= needed

    if int(has_enough.sum()) < 2:
        raise ValueError(
            f"only {int(has_enough.sum())} of {array_rows * array_cols} PEs have "
            f"the {needed} observations needed at sample_size={sample_size}. "
            "Lower --sample-size, or profile with a larger --capacity or "
            "--subsample-size."
        )

    rng = np.random.default_rng(seed)

    # Two disjoint subsamples per PE. Both are drawn here, before any
    # comparison, so that the pooled reference below is built from exactly
    # the same values that each PE is then scored against.
    reference_draws = np.zeros((array_rows, array_cols, sample_size))
    self_draws = np.zeros((array_rows, array_cols, sample_size))

    for row in range(array_rows):
        for col in range(array_cols):
            if not has_enough[row, col]:
                continue

            observed = samples[row, col, : fill[row, col]]
            drawn = rng.choice(observed, size=needed, replace=False)
            reference_draws[row, col] = drawn[:sample_size]
            self_draws[row, col] = drawn[sample_size:]

    # One reference shared by every PE, so the resulting gaps are
    # comparable to each other. Drawing it from the per-PE subsamples rather
    # than from the raw arrays weights every PE equally, regardless of how
    # many observations each happened to record.
    #
    # Each PE contributes to the reference it is then compared against,
    # which pulls its own gap down slightly. The pull is on the order of one
    # part in the number of included PEs, so it is not worth correcting for.
    pooled_candidates = reference_draws[has_enough].ravel()
    pooled = rng.choice(pooled_candidates, size=sample_size, replace=False)

    gaps = np.full((array_rows, array_cols), np.nan)
    self_gaps = np.full((array_rows, array_cols), np.nan)

    for row in range(array_rows):
        for col in range(array_cols):
            if not has_enough[row, col]:
                continue

            gaps[row, col] = ecdf_gap(reference_draws[row, col], pooled)
            self_gaps[row, col] = ecdf_gap(
                reference_draws[row, col], self_draws[row, col]
            )

    return GapGrid(
        gaps=gaps,
        self_gaps=self_gaps,
        sample_size=sample_size,
        included=int(has_enough.sum()),
        total=array_rows * array_cols,
    )


def gap_grids(
    arrays: ProfilingArrays,
    regime: Regime,
    *,
    sample_size: int,
    seed: int = 0,
) -> dict[Variable, GapGrid]:
    """Build a `GapGrid` for every variable `regime` actually records.

    Each variable is compared independently, so a difference in one of them
    stays visible instead of being averaged away against the others.
    """
    grids: dict[Variable, GapGrid] = {}

    for variable in regime.variables_for():
        samples, fill = variable_samples(arrays, regime, variable)
        grids[variable] = gap_grid(samples, fill, sample_size=sample_size, seed=seed)

    return grids
