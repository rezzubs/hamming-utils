"""Tests for fault-injection summary display."""

from .conftest import _make_experiment, _result


def test_fault_summary_off_by_default():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    assert experiment.display().extra() is None


def test_fault_summary_without_compare_bitwise_only_shows_ber_line():
    experiment = _make_experiment(compare_bitwise=False, fault_summary=True, faults=5)
    experiment.run()

    extra = experiment.display().extra()
    assert extra is not None
    assert "Flipped 5/480 bits - BER: 1.04e-02" in extra
    assert "parameters were affected" not in extra
    assert "faulty bit" not in extra


def test_fault_summary_with_compare_bitwise_includes_histogram():
    experiment = _make_experiment(compare_bitwise=True, fault_summary=True, faults=1.0)
    experiment.run()

    extra = experiment.display().extra()
    assert extra is not None
    assert "Flipped 480/480 bits - BER: 1.00e+00" in extra
    assert "15 parameters were affected" in extra
    assert "480 bits were measured faulty (0.00% masked)" in extra
    # Every element gets all 32 bits flipped, so the whole histogram is one bucket.
    assert "15 parameters had 32 faulty bits" in extra


def test_fault_summary_histogram_counts_sum_to_bitmask_length():
    experiment = _make_experiment(compare_bitwise=True, fault_summary=True, faults=200)
    experiment.run()

    bitmask = _result(experiment)["results"][0]["bitmask"]
    extra = experiment.display().extra()
    assert extra is not None

    histogram_lines = [line for line in extra.splitlines() if "faulty bit" in line]
    total = sum(int(line.split()[0]) for line in histogram_lines)
    assert total == len(bitmask)


def test_fault_summary_updates_after_each_run():
    experiment = _make_experiment(compare_bitwise=False, fault_summary=True, faults=3)
    experiment.run()
    first = experiment.display().extra()
    experiment.run()
    second = experiment.display().extra()

    assert first is not None
    assert second is not None
    assert "Flipped 3/480 bits" in first
    assert "Flipped 3/480 bits" in second
