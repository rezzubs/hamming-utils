"""Drawing `profiling_similarity.GapGrid`s as heatmaps over the array.

Figures are built via `matplotlib`'s `Figure` object-oriented API directly
(never `matplotlib.pyplot`), matching `encoded_memory.plots` - see that
module's docstring for why.
"""

import matplotlib
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from systolic.profiling import ProfilingArrays
from systolic.profiling_similarity import (
    GapGrid,
    Regime,
    Variable,
    ecdf_gap_location,
    ecdf_points,
    pooled_sample,
    variable_samples,
)


def build_gap_heatmap_figure(grids: dict[Variable, GapGrid], regime: Regime) -> Figure:
    """One heatmap per variable, each showing how unlike the pooled array
    every PE in a `array_rows x array_cols` grid is.

    Dark/low cells mean a PE looks like the array as a whole; bright/high
    cells mean it doesn't. Horizontal banding says rows differ from each
    other; vertical banding says columns do; a uniformly dark image says
    pooling the whole array is fine. PEs excluded for lack of data (see
    `GapGrid`) are drawn in grey, not black, so "no data" is never
    mistaken for "identical to the array".

    Every panel shares one color scale, from 0 to the largest gap seen in
    any of them, so brightness can be compared across variables directly.

    Raises:
        ValueError: If `grids` is empty.
    """
    if not grids:
        raise ValueError("no grids to plot")

    # One shared scale so a panel isn't drawn brighter than another just
    # because matplotlib stretched its own, smaller range to fill the same
    # color range.
    largest_gap = max(grid.largest_gap() for grid in grids.values())

    variables = list(grids.keys())
    fig = Figure(figsize=(5 * len(variables) + 1, 5))
    fig.set_layout_engine("constrained")
    fig.suptitle(f"{regime.value} regime: per-PE gap against the pooled array")

    mosaic = fig.subplot_mosaic(
        [[*(variable.value for variable in variables), "cbar"]],
        width_ratios=[*([1.0] * len(variables)), 0.06],
    )

    cmap = _colormap_with_grey_for_missing()

    image = None
    for variable in variables:
        ax = mosaic[variable.value]
        assert isinstance(ax, Axes)
        grid = grids[variable]

        image = ax.imshow(grid.gaps, cmap=cmap, vmin=0.0, vmax=largest_gap)
        ax.set_title(_panel_title(variable, grid))
        ax.set_xlabel("PE column")
        ax.set_ylabel("PE row")

    assert image is not None
    colorbar_ax = mosaic["cbar"]
    assert isinstance(colorbar_ax, Axes)
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("gap against pooled array (0 = identical)")

    return fig


def build_pe_ecdf_figure(
    arrays: ProfilingArrays, regime: Regime, *, row: int, col: int
) -> Figure:
    """One PE's ECDF against the pooled array, one panel per variable.

    Shows what `ecdf_gap` actually compares: each panel overlays the PE's
    step function against the pooled one and marks the vertical gap between
    them at its largest point. Use this to look closer at a PE the heatmap
    from `build_gap_heatmap_figure` flagged.

    The gap marked here is computed from every value recorded for the PE
    and the array, not the equal-size subsamples `gap_grid` draws for its
    comparison (see `pooled_sample`) - so it will be close to, but not
    exactly, the heatmap's number for the same PE.

    Raises:
        ValueError: If PE `(row, col)` has no recorded samples in `regime`
            (see `build_gap_heatmap_figure`'s grey cells) - pick a PE with
            data instead.
    """
    variables = regime.variables_for()
    fig = Figure(figsize=(5 * len(variables), 5))
    fig.set_layout_engine("constrained")
    fig.suptitle(f"{regime.value} regime, PE (row={row}, col={col}) vs. pooled array")

    axes = fig.subplots(1, len(variables), squeeze=False)[0]

    for ax, variable in zip(axes, variables, strict=True):
        samples, fill = variable_samples(arrays, regime, variable)
        if fill[row, col] == 0:
            raise ValueError(
                f"PE (row={row}, col={col}) has no recorded {variable.value} "
                f"samples in the {regime.value} regime"
            )
        pe_sample = samples[row, col, : fill[row, col]]
        pooled = pooled_sample(samples, fill)

        _draw_ecdf_overlay(
            ax, pe_sample, pooled, left_label="this PE", right_label="pooled array"
        )
        ax.set_title(variable.value)
        ax.set_xlabel("value")
        ax.set_ylabel("cumulative fraction")
        ax.legend(loc="lower right", fontsize="small")

    return fig


def build_ecdf_spread_figure(arrays: ProfilingArrays, regime: Regime) -> Figure:
    """Every PE's ECDF against the pooled one, one panel per variable.

    Complements `build_gap_heatmap_figure`: the heatmap says *how much* PEs
    differ from the pooled array, this shows *how* - a shifted distribution,
    a heavier tail, a different spread. PEs bunched tightly around the
    pooled curve say pooling the array is fine; PEs fanning away from it say
    the same thing the heatmap's bright cells do, but make the shape of the
    difference visible.
    """
    variables = regime.variables_for()
    fig = Figure(figsize=(5 * len(variables), 5))
    fig.set_layout_engine("constrained")
    fig.suptitle(f"{regime.value} regime: every PE's ECDF against the pooled array")

    axes = fig.subplots(1, len(variables), squeeze=False)[0]

    for ax, variable in zip(axes, variables, strict=True):
        samples, fill = variable_samples(arrays, regime, variable)
        array_rows, array_cols, _capacity = samples.shape

        for row in range(array_rows):
            for col in range(array_cols):
                if fill[row, col] == 0:
                    continue
                x, y = ecdf_points(samples[row, col, : fill[row, col]])
                ax.step(x, y, where="post", color="tab:blue", alpha=0.15, linewidth=0.8)

        pooled_x, pooled_y = ecdf_points(pooled_sample(samples, fill))
        ax.step(
            pooled_x,
            pooled_y,
            where="post",
            color="black",
            linewidth=2,
            label="pooled array",
        )

        ax.set_title(variable.value)
        ax.set_xlabel("value")
        ax.set_ylabel("cumulative fraction")
        ax.legend(loc="lower right", fontsize="small")

    return fig


def _draw_ecdf_overlay(
    ax: Axes,
    left: npt.NDArray[np.floating],
    right: npt.NDArray[np.floating],
    *,
    left_label: str,
    right_label: str,
) -> None:
    """Two ECDF step curves on `ax`, with the largest gap between them marked."""
    left_x, left_y = ecdf_points(left)
    right_x, right_y = ecdf_points(right)
    ax.step(left_x, left_y, where="post", label=left_label)
    ax.step(right_x, right_y, where="post", label=right_label)

    gap_x, gap_left_y, gap_right_y = ecdf_gap_location(left, right)
    gap = abs(gap_left_y - gap_right_y)
    ax.vlines(
        gap_x,
        ymin=min(gap_left_y, gap_right_y),
        ymax=max(gap_left_y, gap_right_y),
        colors="black",
        linestyles="dashed",
    )
    ax.annotate(
        f"gap={gap:.3f}",
        xy=(gap_x, (gap_left_y + gap_right_y) / 2),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize="small",
    )


def _panel_title(variable: Variable, grid: GapGrid) -> str:
    """Panel title naming the variable and, if any PEs were dropped, how many."""
    title = variable.value
    if grid.included < grid.total:
        title += f"\n({grid.included}/{grid.total} PEs had enough data)"
    return title


def _colormap_with_grey_for_missing() -> Colormap:
    """The default sequential colormap, with `NaN` cells (PEs excluded for
    lack of data) drawn in a visible grey rather than left transparent or
    the colormap's own bottom color - either of those would look like a
    real, low-gap PE rather than "no data collected"."""
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad(color="#888888")
    return cmap
