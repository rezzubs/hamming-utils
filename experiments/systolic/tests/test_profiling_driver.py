"""Tests for `run_profiling` (model-level walk + dataset subsampling)."""

from typing import override

import numpy as np
import torch
from faultforge import Fingerprint
from faultforge.dataset import BatchedDataset, DeviceLike
from faultforge.loading import ModelBundle
from faultforge.progress import Progress
from torch import nn
from torch.utils.data import TensorDataset

from systolic.profiling_driver import run_profiling

# `_` prefixed to not interpret it as a Test class.


class _FakeBundle(ModelBundle):
    """A tiny in-memory model/dataset bundle, just enough to drive `run_profiling`.

    `targets` are sorted by class (`index // per_class`), mirroring how
    real datasets (CIFAR/ImageFolder) are typically stored in on-disk class
    order - this is what lets a test catch a subsampler that silently
    reproduces first-N dataset order instead of sampling uniformly.
    """

    def __init__(self, in_features: int, out_features: int, n: int) -> None:
        self._in_features = in_features
        self._out_features = out_features
        self._n = n

    @override
    def load_model(
        self,
        device: DeviceLike,
        *,
        dtype: torch.dtype = torch.float32,
        progress: Progress | None = None,
    ) -> nn.Module:
        _ = progress
        return nn.Sequential(
            nn.Linear(self._in_features, 6),
            nn.Linear(6, self._out_features),
        ).to(device=device, dtype=dtype)

    @override
    def load_dataset(
        self,
        batch_size: int,
        device: DeviceLike,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        progress: Progress | None = None,
    ) -> BatchedDataset:
        _ = progress
        inputs = torch.randn(self._n, self._in_features)
        per_class = max(1, self._n // self._out_features)
        targets = torch.arange(self._n) // per_class
        dataset = TensorDataset(inputs, targets)
        return BatchedDataset.from_dataset(
            dataset, batch_size, device, shuffle=shuffle, seed=seed
        )

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="fake_bundle")


def test_run_profiling_settled_shapes() -> None:
    nrows, ncols, capacity = 4, 4, 8
    bundle = _FakeBundle(in_features=4, out_features=3, n=40)

    artifact, metadata = run_profiling(
        bundle,
        (nrows, ncols),
        capacity,
        subsample_size=8,
        seed=0,
    )

    assert artifact.first_triples.shape == (nrows, ncols, capacity, 3)
    assert artifact.first_fill.shape == (nrows, ncols)
    assert artifact.active_triples.shape == (nrows, ncols, capacity, 3)
    assert artifact.drain_partial_sums.shape == (nrows, ncols, capacity)
    assert metadata.array_rows == nrows
    assert metadata.array_cols == ncols
    assert metadata.capacity == capacity
    assert metadata.seed == 0


def test_run_profiling_subsample_is_randomized_not_first_n() -> None:
    """Regression test for the bug this chunk's core change fixes: without
    `shuffle=True`, subsampling would just take the dataset's first N items,
    which for a class-sorted dataset means profiling only ever sees one
    class - exactly what "uniform random, not stratified" rules out.
    """
    nrows, ncols, capacity = 4, 4, 10_000
    out_features = 5
    bundle = _FakeBundle(in_features=4, out_features=out_features, n=100)

    artifact, _ = run_profiling(
        bundle,
        (nrows, ncols),
        capacity,
        subsample_size=20,
        seed=0,
    )

    # A single class occupies one contiguous 20-item block of the sorted
    # dataset (100 items / 5 classes). If the subsample were really "the
    # first 20 items" (no shuffle), every profiled input would come from
    # class 0 alone, and a different seed would make no difference.
    other_seed_artifact, _ = run_profiling(
        bundle,
        (nrows, ncols),
        capacity,
        subsample_size=20,
        seed=1,
    )
    assert not np.array_equal(
        artifact.active_triples, other_seed_artifact.active_triples
    )


def test_run_profiling_subsample_clamps_to_dataset_size() -> None:
    nrows, ncols, capacity = 4, 4, 100
    bundle = _FakeBundle(in_features=4, out_features=3, n=10)

    artifact, _ = run_profiling(
        bundle,
        (nrows, ncols),
        capacity,
        subsample_size=1000,
        seed=0,
    )

    assert artifact.active_fill.sum() > 0
