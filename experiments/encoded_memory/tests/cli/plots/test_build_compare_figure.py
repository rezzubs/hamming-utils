"""Tests for `encoded_memory.plots.build_compare_figure`."""

import pytest
from encoded_memory import ReliabilityMetric
from encoded_memory.plots import GroupBy, build_compare_figure
from matplotlib.figure import Figure


def test_build_compare_figure_returns_figure(tmp_path, make_configuration):
    config_a = make_configuration(tmp_path / "a", label="A")
    config_b = make_configuration(tmp_path / "b", label="B")
    fig = build_compare_figure([config_a, config_b])
    assert isinstance(fig, Figure)


def test_build_compare_figure_grid_by_model(tmp_path, make_configuration):
    config_a = make_configuration(tmp_path / "a", label="A", model="resnet18")
    config_b = make_configuration(tmp_path / "b", label="B", model="resnet50")
    fig = build_compare_figure([config_a, config_b], col_by=GroupBy.Model)
    assert isinstance(fig, Figure)


def test_build_compare_figure_raises_on_empty():
    with pytest.raises(ValueError, match="no configurations"):
        build_compare_figure([])


def test_build_compare_figure_raises_on_mismatched_metric(tmp_path, make_configuration):
    sdc = make_configuration(
        tmp_path / "sdc", label="sdc", metric=ReliabilityMetric.Sdc
    )
    accuracy = make_configuration(tmp_path / "acc", label="accuracy")

    with pytest.raises(ValueError, match="reliability metric"):
        build_compare_figure([sdc, accuracy])
