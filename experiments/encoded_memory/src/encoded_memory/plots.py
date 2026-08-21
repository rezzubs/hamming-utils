"""Figure-building for the `compare`/`heatmap` commands.

See `results` for the vocabulary (configuration, bit error rate, split key) and
the `compare` pipeline this is the final step of. Figures are built via
`matplotlib`'s `Figure` object-oriented API directly (never
`matplotlib.pyplot`), so this module never touches `pyplot`'s global "current
figure" state; only `commands` imports `pyplot`, for `plt.show()`.

`matplotlib` is loosely typed - several of its calls hand back `Any` - so
extracted axes are pinned to `Axes` with `assert isinstance` at the boundary,
which keeps editor tooling useful in the rest of the function.
"""

import enum
import logging
from collections.abc import Sequence
from typing import Any

import matplotlib
import numpy as np
from encoded_memory import (
    DetailedResult,
    ReliabilityMetric,
    SavedResult,
)
from encoded_memory.results import (
    Configuration,
    bit_position_histogram,
    configuration_points,
)
from faultforge import Fingerprint
from faultforge.dtype import EncodingDtype
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

_MARKERS = [
    "o",
    "v",
    "^",
    "<",
    ">",
    "s",
    "p",
    "P",
    "*",
    "h",
    "+",
    "X",
    "D",
]
"""A fixed marker cycle so lines stay distinguishable without relying on color alone."""

_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
"""Matplotlib's default ("tab10") color cycle, assigned explicitly per label
(see `build_compare_figure`) rather than left to each `Axes`'s own color
cycler - the cycler resets per `Axes`, so two configurations plotted into
different, otherwise-empty grid cells would both land on its first color."""


class GroupBy(enum.StrEnum):
    """A fingerprint-derived split key to fan a `compare` grid out by.

    Distinct from a configuration (see `results`): this decides which grid
    *cell* a line lands in, not which line results belong to.
    """

    Model = "model"
    Dtype = "dtype"
    Dataset = "dataset"
    Metric = "metric"
    Ungrouped = "none"
    """No splitting - a single row/column. `None` can't be a member name (it's
    a reserved keyword), but the CLI-facing value stays "none"."""


def group_key(group_by: GroupBy, fingerprint: Fingerprint) -> str | None:
    """The grid cell value `fingerprint` belongs to under `group_by`.

    `None` for `GroupBy.Ungrouped`.

    Raises:
        ValueError: For `GroupBy.Dataset` on a bundle with no `dataset`
            scalar (e.g. an `ImageNet` bundle).
    """
    match group_by:
        case GroupBy.Ungrouped:
            return None
        case GroupBy.Dtype:
            return str(fingerprint.scalars["dtype"])
        case GroupBy.Metric:
            return str(fingerprint.scalars["reliability_metric"])
        case GroupBy.Model:
            return str(fingerprint.children["bundle"][0].scalars["model"])
        case GroupBy.Dataset:
            bundle = fingerprint.children["bundle"][0]
            dataset = bundle.scalars.get("dataset")
            if dataset is None:
                raise ValueError(
                    f"{bundle.kind!r} bundle has no 'dataset' scalar to group by"
                )
            return str(dataset)


def _reliability_metric(configuration: Configuration) -> ReliabilityMetric:
    _, first = configuration.results[0]
    return first.reliability_metric()


def _cell(axes: Any, row: int, col: int) -> Axes:
    """One grid cell out of `Figure.subplots`' loosely-typed 2D array."""
    ax = axes[row][col]
    assert isinstance(ax, Axes)
    return ax


def build_compare_figure(
    configurations: Sequence[Configuration],
    *,
    row_by: GroupBy = GroupBy.Ungrouped,
    col_by: GroupBy = GroupBy.Ungrouped,
    percentile: float | None = None,
    log_x: bool = False,
) -> Figure:
    """Step 4 of the `compare` pipeline: one subplot per `(row_by, col_by)`
    cell, one overlaid line per configuration within its cell.

    Cell values are discovered in first-seen order across `configurations`,
    sizing the grid automatically. A label recurring across several
    configurations (e.g. the same encoder compared across several model/dtype
    combinations) keeps the same marker/color wherever it appears, and only
    contributes one entry to the shared figure-level legend.

    Raises:
        ValueError: If `configurations` is empty, if they don't all share a
            `reliability_metric` (unlike model/dataset/dtype, it changes what
            the y-axis numbers mean, not just what's being compared), or if
            `row_by`/`col_by` raises (see `group_key`).
    """
    if not configurations:
        raise ValueError("no configurations to plot")

    metrics = {_reliability_metric(config) for config in configurations}
    if len(metrics) > 1:
        names = sorted(metric.value for metric in metrics)
        raise ValueError(
            f"all configurations must share a reliability metric to be "
            f"compared, got {names}"
        )
    metric = next(iter(metrics))

    row_values: list[str | None] = []
    col_values: list[str | None] = []
    cells: list[tuple[str | None, str | None]] = []
    for config in configurations:
        fingerprint = config.results[0][1].fingerprint
        row_value = group_key(row_by, fingerprint)
        col_value = group_key(col_by, fingerprint)
        cells.append((row_value, col_value))
        if row_value not in row_values:
            row_values.append(row_value)
        if col_value not in col_values:
            col_values.append(col_value)

    n_rows = len(row_values)
    n_cols = len(col_values)

    labels_in_order: list[str] = []
    for config in configurations:
        if config.label not in labels_in_order:
            labels_in_order.append(config.label)
    marker_by_label = {
        label: _MARKERS[index % len(_MARKERS)]
        for index, label in enumerate(labels_in_order)
    }
    color_by_label = {
        label: _COLORS[index % len(_COLORS)]
        for index, label in enumerate(labels_in_order)
    }

    fig = Figure(figsize=(4 * n_cols + 2, 3 * n_rows + 1))
    fig.set_layout_engine("constrained")
    axes = fig.subplots(n_rows, n_cols, squeeze=False, sharex=True, sharey=True)

    handles_by_label: dict[str, Line2D] = {}
    for config, (row_value, col_value) in zip(configurations, cells, strict=True):
        ax = _cell(axes, row_values.index(row_value), col_values.index(col_value))

        points = configuration_points(config, percentile=percentile)
        xs = [x for x, _ in points]
        ys = [y for _, y in points]

        (line,) = ax.plot(
            xs,
            ys,
            label=config.label,
            marker=marker_by_label[config.label],
            color=color_by_label[config.label],
        )
        handles_by_label.setdefault(config.label, line)

        if log_x:
            ax.set_xscale("log")
        ax.grid(True)

    # Label each row/column of the grid with its split-key value (the key name
    # itself is carried by the axis titles below).
    for row_index, row_value in enumerate(row_values):
        if row_value is not None:
            _cell(axes, row_index, 0).set_ylabel(row_value)
    for col_index, col_value in enumerate(col_values):
        if col_value is not None:
            _cell(axes, 0, col_index).set_title(col_value)

    score_label = "Mean" if percentile is None else f"{percentile:g}th Percentile"
    fig.supxlabel("Bit Error Rate")
    fig.supylabel(f"{score_label} {metric.score_name()} [%]")

    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        loc="outside upper left",
        frameon=False,
        ncols=2,
    )

    return fig


def _residual_faults(run_bitmask: Sequence[int]) -> int:
    """Total bits that differ from golden post-decode, across one run.

    The count of bits actually flipped after decoding, which can be less than
    the injected fault count if the encoding masked some of them.
    """
    return sum(value.bit_count() for value in run_bitmask)


def _histogram_range(values: Sequence[float]) -> tuple[float, float]:
    """A valid (non-zero-width) range for `numpy.histogram2d`."""
    low = min(values)
    high = max(values)
    return (low, high) if low != high else (low, low + 1)


def _colorbar_ticks(vmax: float) -> list[int]:
    """Positive integer occurrence counts up to `vmax`, in the standard
    "1-2-5" preferred-number sequence (1, 2, 5, 10, 20, 50, ...) - the same
    convention `LogLocator(subs=[1, 2, 5])` uses for log-scaled axes.

    Explicit rather than left to `LogNorm`'s default locator: occurrence
    counts are always positive integers, but when a panel's real range is
    narrow (e.g. every visible cell has exactly one occurrence), that locator
    invents fractional, decade-boundary ticks outside the real data (like
    `10^-1`, i.e. 0.1 occurrences) to pad out a nonsingular range - which
    reads as a negative value and doesn't correspond to anything real.
    """
    ticks: list[int] = []
    magnitude = 1
    while magnitude <= vmax:
        for multiplier in (1, 2, 5):
            tick = magnitude * multiplier
            if tick <= vmax:
                ticks.append(tick)
        magnitude *= 10
    return ticks if ticks else [1]


def build_heatmap_figure(
    results: Sequence[SavedResult],
    *,
    bins: int = 50,
    min_score: float | None = None,
    max_score: float | None = None,
    max_total_faults: int | None = None,
    skip_multi_bit_faults: bool = False,
) -> Figure:
    """A figure of per-run score density against faulty bit position (built
    from `bit_position_histogram`) - shows which bit positions correlate with
    reliability loss.

    Bins with no data are filled with the colormap's minimum color rather
    than left transparent - `LogNorm` treats a zero count as invalid and
    masks it, which would otherwise show through as plain white.

    Raises:
        ValueError: If `results` is empty, any entry isn't a `DetailedResult`
            (a heatmap fundamentally needs per-run `bitmask` data), they don't
            all share a `dtype`/`reliability_metric` (both affect axis
            semantics/bounds, not just cosmetics), or no runs remain after
            filtering.
    """
    if not results:
        raise ValueError("no results to plot")

    for result in results:
        if not isinstance(result.result, DetailedResult):
            raise ValueError(
                "heatmap requires results recorded with --compare-bitwise "
                "(a per-run bitmask)"
            )

    metrics = {result.reliability_metric() for result in results}
    if len(metrics) > 1:
        names = sorted(metric.value for metric in metrics)
        raise ValueError(f"all results must share a reliability metric, got {names}")
    metric = next(iter(metrics))

    dtypes = {result.fingerprint.scalars["dtype"] for result in results}
    if len(dtypes) > 1:
        raise ValueError(f"all results must share a dtype, got {sorted(dtypes)}")
    bit_width = EncodingDtype(next(iter(dtypes))).bit_count()

    scores: list[float] = []
    positions: list[int] = []
    weights: list[int] = []

    for result in results:
        assert isinstance(result.result, DetailedResult)
        for run, score in zip(result.result.results, result.scores(), strict=True):
            if min_score is not None and score < min_score:
                continue
            if max_score is not None and score > max_score:
                continue

            if max_total_faults is not None:
                residual = _residual_faults(run.bitmask)
                if residual > max_total_faults:
                    continue

            histogram = bit_position_histogram(
                run.bitmask, skip_multi_bit=skip_multi_bit_faults
            )
            for position, count in histogram.items():
                scores.append(score)
                positions.append(position)
                weights.append(count)

    if not scores:
        raise ValueError("no runs left after filtering")

    score_range = _histogram_range(scores)
    x_edges = np.histogram_bin_edges([], bins=bins, range=score_range)

    counts, _, y_edges = np.histogram2d(
        scores,
        positions,
        bins=[x_edges, bit_width],
        range=[score_range, (-0.5, bit_width - 0.5)],
        weights=weights,
    )

    cmap = matplotlib.colormaps["plasma"]
    background = cmap(0.0)
    # `vmin` is deliberately below the smallest real occurrence count (1),
    # not equal to it - `LogNorm` maps `vmin` itself to the very bottom of
    # the colormap, which is also `background` above; if a real count of 1
    # used exactly `vmin`, it would render identically to an empty cell.
    norm = LogNorm(vmin=0.5, vmax=counts.max())

    fig = Figure(figsize=(8, 5))
    fig.set_layout_engine("constrained")
    mosaic = fig.subplot_mosaic([["main", "cbar"]], width_ratios=[0.95, 0.05])
    ax = mosaic["main"]
    colorbar_ax = mosaic["cbar"]
    assert isinstance(ax, Axes)
    assert isinstance(colorbar_ax, Axes)
    ax.set_facecolor(background)

    image = ax.pcolormesh(x_edges, y_edges, counts.T, cmap=cmap, norm=norm)
    ax.set_xlabel(f"{metric.score_name()} [%]")
    ax.set_ylabel("Bit Position")

    colorbar = fig.colorbar(image, cax=colorbar_ax, ticks=_colorbar_ticks(counts.max()))
    # `LogNorm` colorbars auto-attach sub-decade minor ticks (e.g. `6x10^-1`)
    # independent of the `ticks=` override above - turn them off so only the
    # explicit, always-meaningful integer ticks show.
    colorbar.minorticks_off()
    colorbar_ax.set_ylabel("Occurrences")

    return fig
