"""Tests for `encoded_memory.results.configuration_points`."""

from encoded_memory.results import configuration_points


def test_configuration_points_pools_and_sorts_by_rate(tmp_path, make_configuration):
    configuration = make_configuration(tmp_path / "a", label="a", rates=(0.05, 0.01))
    points = configuration_points(configuration, percentile=None)
    rates = [rate for rate, _ in points]
    assert rates == sorted(rates)
    assert len(points) == 2
