"""Tests for `encoded_memory.plots.build_heatmap_figure`."""

import pytest
from encoded_memory.plots import build_heatmap_figure
from encoded_memory.results import load_results
from matplotlib.figure import Figure


def test_build_heatmap_figure_returns_figure(tmp_path, make_configuration):
    configuration = make_configuration(tmp_path / "a", label="a")
    results = [result for _, result in configuration.results]
    fig = build_heatmap_figure(results)
    assert isinstance(fig, Figure)


def test_build_heatmap_figure_raises_on_empty():
    with pytest.raises(ValueError, match="no results"):
        build_heatmap_figure([])


def test_build_heatmap_figure_raises_without_compare_bitwise(tmp_path, save_result):
    save_result(tmp_path / "simple.json", faults=0.05, compare_bitwise=False)
    loaded = load_results([tmp_path])
    results = [result for _, result in loaded]

    with pytest.raises(ValueError, match="compare-bitwise"):
        build_heatmap_figure(results)
