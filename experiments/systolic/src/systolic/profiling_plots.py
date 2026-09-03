"""Drawing `profiling_similarity.GapGrid`s as heatmaps over the array.

Figures are built via `matplotlib`'s `Figure` object-oriented API directly
(never `matplotlib.pyplot`), matching `encoded_memory.plots` - see that
module's docstring for why.
"""

import matplotlib
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from systolic.profiling_similarity import GapGrid, Regime, Variable


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
