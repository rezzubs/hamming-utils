"""Abstract interface for loading a model together with its dataset."""

import abc

import torch
from torch import nn

from faultforge._internal.dataset import BatchedDataset, DeviceLike
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.progress import Progress

DEFAULT_DTYPE: torch.dtype = torch.float32


class ModelBundle(abc.ABC):
    """A type which can load a model and its associated dataset"""

    @abc.abstractmethod
    def load_model(
        self,
        device: DeviceLike,
        *,
        dtype: torch.dtype = DEFAULT_DTYPE,
        progress: Progress | None = None,
    ) -> nn.Module:
        """Load the model, casting its parameters to `dtype`."""

    @abc.abstractmethod
    def load_dataset(
        self, batch_size: int, device: DeviceLike, *, progress: Progress | None = None
    ) -> BatchedDataset:
        """Load the dataset."""

    @abc.abstractmethod
    def fingerprint(self) -> Fingerprint:
        """Return a structural identity for this model and dataset.

        Only the parts that change which model and dataset are loaded belong in
        the fingerprint; environmental details like the filesystem cache or
        download location are left out.
        """
