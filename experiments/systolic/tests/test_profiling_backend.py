"""Tests for `ProfilingBackend`."""

import pytest
import torch

from systolic.profiling_backend import ProfilingBackend


def test_profiling_backend_rejects_non_float32() -> None:
    backend = ProfilingBackend(nrows=4, ncols=4, capacity=8, seed=0)
    weights = torch.randn(3, 4, dtype=torch.float64)
    activations = torch.randn(4, 2, dtype=torch.float64)

    with pytest.raises(ValueError, match="float32"):
        backend.matmul(weights, activations)


def test_profiling_backend_set_fault_raises() -> None:
    backend = ProfilingBackend(nrows=4, ncols=4, capacity=8, seed=0)
    with pytest.raises(NotImplementedError):
        backend.set_fault(None)


def test_profiling_backend_reports_array_size() -> None:
    backend = ProfilingBackend(nrows=4, ncols=6, capacity=8, seed=0)
    assert backend.nrows() == 4
    assert backend.ncols() == 6


def test_profiling_backend_accumulates_across_calls() -> None:
    """Two `.matmul()` calls (standing in for two layers, or one layer over
    two batches) both contribute to `.artifact()` - not just the last one.
    """
    nrows, ncols, capacity, seed = 4, 4, 1000, 0
    weights = torch.randn(3, 4)
    activations_a = torch.randn(4, 2)
    activations_b = torch.randn(4, 2)

    baseline_a = ProfilingBackend(nrows, ncols, capacity, seed)
    baseline_a.matmul(weights, activations_a)

    baseline_b = ProfilingBackend(nrows, ncols, capacity, seed)
    baseline_b.matmul(weights, activations_b)

    combined = ProfilingBackend(nrows, ncols, capacity, seed)
    combined.matmul(weights, activations_a)
    combined.matmul(weights, activations_b)

    artifact_a = baseline_a.artifact()
    artifact_b = baseline_b.artifact()
    artifact_combined = combined.artifact()

    assert artifact_combined.active_fill.sum() == (
        artifact_a.active_fill.sum() + artifact_b.active_fill.sum()
    )
