"""Tests for the `Profiler`/`ProfilingArtifact` bindings.

Also covers the `systolic.profiling` save/load round trip.
"""

import numpy as np

from systolic._rust import Mapping, Profiler
from systolic.profiling import (
    ProfilingMetadata,
    load_profiling_artifact,
    save_profiling_artifact,
)

_ARRAY_NAMES = {
    "first_triples",
    "first_fill",
    "active_triples",
    "active_fill",
    "drain_partial_sums",
    "drain_fill",
    "metadata_json",
}


def test_profiler_accumulates_across_runs() -> None:
    """A `Profiler` fed two batches ends up with the pooled fill counts of both.

    Not just the last one - this is the property a one-shot profiling
    function (construct a hook, run once, extract immediately) cannot
    express, since it would discard everything between calls.
    """
    rng = np.random.default_rng(0)
    out_features, in_features, batch = 3, 4, 2
    array_nrows, array_ncols = in_features, out_features
    capacity, seed = 8, 0

    weights = rng.standard_normal((out_features, in_features)).astype(np.float32)
    mapping = Mapping.auto_for(weights, array_nrows, array_ncols)
    activations_a = rng.standard_normal((in_features, batch)).astype(np.float32)
    activations_b = rng.standard_normal((in_features, batch)).astype(np.float32)

    baseline_a = Profiler(array_nrows, array_ncols, capacity, seed)
    baseline_a.run(mapping, weights, activations_a)
    artifact_a = baseline_a.artifact()

    baseline_b = Profiler(array_nrows, array_ncols, capacity, seed)
    baseline_b.run(mapping, weights, activations_b)
    artifact_b = baseline_b.artifact()

    combined = Profiler(array_nrows, array_ncols, capacity, seed)
    combined.run(mapping, weights, activations_a)
    combined.run(mapping, weights, activations_b)
    artifact_combined = combined.artifact()

    # Capacity (8) comfortably exceeds either run's per-PE observation count
    # (batch=2 per regime per run), so no reservoir eviction occurs and fill
    # counts are exactly additive, independent of the shared RNG stream.
    np.testing.assert_array_equal(
        artifact_combined.first_fill, artifact_a.first_fill + artifact_b.first_fill
    )
    np.testing.assert_array_equal(
        artifact_combined.active_fill, artifact_a.active_fill + artifact_b.active_fill
    )
    np.testing.assert_array_equal(
        artifact_combined.drain_fill, artifact_a.drain_fill + artifact_b.drain_fill
    )


def test_profiling_artifact_round_trips(tmp_path) -> None:
    rng = np.random.default_rng(0)
    out_features, in_features, batch = 3, 4, 2
    array_nrows, array_ncols = in_features, out_features
    capacity, seed = 8, 0

    weights = rng.standard_normal((out_features, in_features)).astype(np.float32)
    activations = rng.standard_normal((in_features, batch)).astype(np.float32)
    mapping = Mapping.auto_for(weights, array_nrows, array_ncols)

    profiler = Profiler(array_nrows, array_ncols, capacity, seed)
    profiler.run(mapping, weights, activations)
    artifact = profiler.artifact()

    # Pin the settled storage-format shapes down at the source, before even
    # touching disk.
    assert artifact.first_triples.shape == (array_nrows, array_ncols, capacity, 3)
    assert artifact.first_fill.shape == (array_nrows, array_ncols)
    assert artifact.active_triples.shape == (array_nrows, array_ncols, capacity, 3)
    assert artifact.drain_partial_sums.shape == (array_nrows, array_ncols, capacity)

    # At least one PE must show real ACTIVE fill for the padding check
    # below to be meaningful.
    active_pe = next(
        (y, x)
        for y in range(array_nrows)
        for x in range(array_ncols)
        if artifact.active_fill[y, x] > 0
    )
    fill = artifact.active_fill[active_pe]
    assert fill < capacity, "test needs unfilled padding to check against"
    np.testing.assert_array_equal(
        artifact.active_triples[active_pe[0], active_pe[1], fill:, :], 0.0
    )

    metadata = ProfilingMetadata(
        array_rows=array_nrows, array_cols=array_ncols, capacity=capacity, seed=seed
    )

    path = tmp_path / "artifact.npz"
    save_profiling_artifact(path, artifact, metadata)

    # A plain `np.load` must see exactly the six real arrays plus the
    # embedded metadata array - no extra or missing members.
    with np.load(path) as npz:
        assert set(npz.files) == _ARRAY_NAMES

    loaded_arrays, loaded_metadata = load_profiling_artifact(path)

    np.testing.assert_array_equal(loaded_arrays.first_triples, artifact.first_triples)
    np.testing.assert_array_equal(loaded_arrays.first_fill, artifact.first_fill)
    np.testing.assert_array_equal(loaded_arrays.active_triples, artifact.active_triples)
    np.testing.assert_array_equal(loaded_arrays.active_fill, artifact.active_fill)
    np.testing.assert_array_equal(
        loaded_arrays.drain_partial_sums, artifact.drain_partial_sums
    )
    np.testing.assert_array_equal(loaded_arrays.drain_fill, artifact.drain_fill)
    assert loaded_metadata == metadata
