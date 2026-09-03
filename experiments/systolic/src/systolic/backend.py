"""Abstract base class for systolic-array backends.

See `systolic` for a general overview.
"""
from typing import final

import abc

import torch
from torch import Tensor

from systolic._rust import Fault, Mapping


class SystolicBackend(abc.ABC):
    """A weight-stationary systolic-array evaluation strategy.

    `weights`/`activations` name the physical weight-stationary roles that
    fault-lifting recipes for this array depend on, and the hardware convention
    (not PyTorch's) applies to the shapes.
    """

    @abc.abstractmethod
    def matmul(self, weights: Tensor, activations: Tensor) -> Tensor:
        """Compute `weights @ activations`.

        `weights` has shape `(out_features, in_features)`, `activations` has
        shape `(in_features, batch)`. Returns `(out_features, batch)`.
        """
        ...

    @abc.abstractmethod
    def set_fault(self, fault: Fault | None) -> None:
        """Set (or clear, with `None`) the fault applied by `matmul`."""
        ...

    @abc.abstractmethod
    def nrows(self) -> int:
        """Return the number of rows in the systolic array."""
        ...

    @abc.abstractmethod
    def ncols(self) -> int:
        """Return the number of columns in the systolic array."""
        ...


@final
class MappingCache:
    """Caches one `Mapping` per distinct weight shape for a fixed array size.

    A physical fault lifts/simulates differently per layer, but the mapping
    itself only depends on the weight shape and array size - backends with
    many layers sharing one array reuse the same cache instance across all
    of them.
    """

    def __init__(self, nrows: int, ncols: int) -> None:
        self._nrows = nrows
        self._ncols = ncols
        self._mappings: dict[torch.Size, Mapping] = {}

    def get(self, weights: Tensor) -> Mapping:
        """Return the cached `Mapping` for `weights.shape`, building it if needed."""
        mapping = self._mappings.get(weights.shape)
        if mapping is None:
            mapping = Mapping.auto_for(
                weights.numpy(force=True), self._nrows, self._ncols
            )
            self._mappings[weights.shape] = mapping
        return mapping
