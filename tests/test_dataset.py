"""Tests for faultforge._internal.dataset (BatchedDataset.batch_count, CachedDataset)."""

import math
from typing import override

import pytest
import torch
from faultforge._internal.dataset import BatchedDataset
from faultforge._internal.progress import ProgressStage
from faultforge.progress import Progress
from torch import Tensor
from torch.utils.data import Dataset

# The following classes are `_` prefixed to not interpret them as Test classes.


class _SizedDataset(Dataset):
    """A map-style dataset that reports its length."""

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return torch.tensor([float(index)]), torch.tensor([0])


class _UnsizedDataset(Dataset):
    """A map-style dataset that does not implement `__len__`."""

    def __init__(self, n: int) -> None:
        self._n = n

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index >= self._n:
            raise IndexError
        return torch.tensor([float(index)]), torch.tensor([0])


def test_batch_count_known_length() -> None:
    dataset = BatchedDataset.from_dataset(_SizedDataset(10), batch_size=3)
    assert dataset.batch_count() == math.ceil(10 / 3)


def test_batch_count_unknown_length() -> None:
    dataset = BatchedDataset.from_dataset(_UnsizedDataset(10), batch_size=3)
    assert dataset.batch_count() is None


def test_cached_dataset_batch_count() -> None:
    dataset = BatchedDataset.from_dataset(_SizedDataset(10), batch_size=3)
    cached = dataset.precompute()
    assert cached.batch_count() == math.ceil(10 / 3)


def test_precompute_calls_advance_once_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = ProgressStage.advance

    def counting_advance(self: ProgressStage, n: int = 1) -> None:
        calls.append(n)
        original(self, n)

    monkeypatch.setattr(ProgressStage, "advance", counting_advance)

    dataset = BatchedDataset.from_dataset(_SizedDataset(5), batch_size=1)
    _ = dataset.precompute(progress=Progress())

    assert calls == [1] * 5
