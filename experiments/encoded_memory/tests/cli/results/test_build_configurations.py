"""Tests for `encoded_memory.results.build_configurations`."""

from encoded_memory import ReliabilityMetric
from encoded_memory.results import build_configurations, load_results


def test_build_configurations_merges_matching_fingerprints_across_paths(
    tmp_path, save_result
):
    for faults, directory in [(1, "a"), (2, "a"), (3, "b")]:
        save_result(tmp_path / directory / f"faults_{faults}.json", faults=faults)

    loaded = load_results([tmp_path])
    configurations = build_configurations(loaded, label_overrides={})

    # All three share every fingerprint scalar except `faults`, so they merge
    # into one configuration regardless of which directory they came from.
    assert len(configurations) == 1
    assert len(configurations[0].results) == 3


def test_build_configurations_separates_differing_configurations(tmp_path, save_result):
    save_result(tmp_path / "a.json", faults=1)
    save_result(tmp_path / "b.json", faults=2, metric=ReliabilityMetric.Sdc)

    loaded = load_results([tmp_path])
    configurations = build_configurations(loaded, label_overrides={})

    assert len(configurations) == 2


def test_build_configurations_label_override_beats_filename_default(
    tmp_path, save_result
):
    path = save_result(tmp_path / "ber_0.05.json", faults=0.05)

    loaded = load_results([tmp_path])
    (default,) = build_configurations(loaded, label_overrides={})
    assert default.label == "ber_0.05"

    (overridden,) = build_configurations(loaded, label_overrides={path: "Identity"})
    assert overridden.label == "Identity"
