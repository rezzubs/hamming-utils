"""Types related to datasets."""

import abc
import logging
import math
from collections.abc import Iterable, Sized
from dataclasses import dataclass
from typing import Any, Iterator, Self, final, override

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from faultforge._internal.progress import Progress, stage

logger = logging.getLogger(__name__)

type DeviceLike = torch.device | str | int

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_BATCH_SIZE: int = 256


@dataclass(slots=True)
class DataBatch:
    """A single batch of data containing input tensors and targets."""

    inputs: Tensor
    """A batch of input images."""
    targets: Tensor
    """A batch of expected outputs."""

    def to(self, device: DeviceLike) -> Self:
        """Moves the batch to the specified device.

        This is done in-place.
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)

        return self


class BatchedDataset(abc.ABC):
    """An iterator over batches of image data.

    Provides a type safe API over `torch.utils.data.DataLoader`.

    Use `from_dataset` to create a `BatchedDataset` from any existing dataset.
    """

    @abc.abstractmethod
    def __next__(self) -> DataBatch:
        """Returns the next batch of data."""

    @abc.abstractmethod
    def batch_size(self) -> int:
        """Return the batch size of this dataset."""

    @abc.abstractmethod
    def to(self, device: DeviceLike) -> Self:
        """Maps all future batches to the specified device."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Start iteration from the beginning."""

    def batch_count(self) -> int | None:
        """Return the total number of batches, or `None` if unknowable ahead of time."""
        return None

    def __iter__(self) -> Self:
        return self

    @staticmethod
    def from_dataset(
        dataset: Dataset[Any],
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: DeviceLike = DEFAULT_DEVICE,
    ) -> BatchedDataset:
        return _BatchedDataset(
            dataset,
            torch.device(device),
            batch_size,
        )

    def precompute(
        self, limit: int | None = None, *, progress: Progress | None = None
    ) -> CachedDataset:
        """Precompute all batches.

        See `CachedDataset` for details.
        """
        return CachedDataset(self, limit, progress=progress)


@final
@dataclass(slots=True)
class _BatchedDataset(BatchedDataset):
    _dataset: Dataset[Any]
    _loader: Iterator[Any]
    _device: torch.device
    _batch_size: int

    def __init__(
        self, dataset: Dataset[Any], device: torch.device, batch_size: int
    ) -> None:
        self._dataset = dataset
        self._device = device
        self._batch_size = batch_size
        self._loader = self._get_loader()
        self.reset()

    def _get_loader(self) -> Iterator[Any]:
        return iter(
            DataLoader(
                self._dataset,
                batch_size=self._batch_size,
                shuffle=False,
            )
        )

    @override
    def __next__(self) -> DataBatch:
        batch = next(self._loader)

        if not isinstance(batch, Iterable):
            raise TypeError(
                f"Expected next(dataloader) to return an instance of `Iterable`, got {type(batch)}"
            )

        batch = list(batch)
        if len(batch) != 2:
            raise ValueError(
                f"Expected the dataset to yield two items, got {len(batch)}"
            )

        [inputs, labels] = batch

        if not isinstance(inputs, Tensor):
            raise TypeError(f"expected inputs to be a Tensor, got {type(inputs)}")
        if not isinstance(labels, Tensor):
            raise TypeError(f"expected labels to be a Tensor, got {type(labels)}")

        return DataBatch(inputs, labels).to(self._device)

    @override
    def to(self, device: DeviceLike) -> Self:
        self._device = torch.device(device)
        return self

    @override
    def batch_size(self) -> int:
        return self._batch_size

    @override
    def reset(self) -> None:
        self._loader = self._get_loader()

    @override
    def batch_count(self) -> int | None:
        if not isinstance(self._dataset, Sized):
            return None
        return math.ceil(len(self._dataset) / self._batch_size)


@final
@dataclass(slots=True)
class CachedDataset(BatchedDataset):
    """A dataset that precomputes all batches and keeps them in memory.

    This is useful when the dataset needs to be used multiple times and the
    overhead of computing the batches is significant. As a rule of thumb, you
    should almost definitely use this over a normal `BatchedDataset` if it's
    feasible to store the full data in memory.
    """

    cursor: int
    _items: list[DataBatch]
    _batch_size: int

    def __init__(
        self,
        dataset: BatchedDataset,
        limit: int | None = None,
        *,
        progress: Progress | None = None,
    ) -> None:
        self._batch_size = dataset.batch_size()
        self._items = []

        total = dataset.batch_count()
        if limit is not None:
            total = limit if total is None else min(total, limit)

        with stage(progress, "Precomputing dataset batches", total=total) as s:
            for i, batch in enumerate(dataset):
                if limit is not None and i >= limit:
                    break
                self._items.append(batch)
                s.advance()

        self.cursor = 0

    @override
    def reset(self) -> None:
        """Enables iteration from the beginning"""
        self.cursor = 0

    @override
    def batch_size(self) -> int:
        return self._batch_size

    @override
    def batch_count(self) -> int | None:
        return len(self._items)

    @override
    def to(self, device: DeviceLike) -> Self:
        for item in self._items:
            item.to(device)
        return self

    @override
    def __next__(self) -> DataBatch:
        if self.cursor >= len(self._items):
            raise StopIteration
        batch = self._items[self.cursor]
        self.cursor += 1
        return batch
