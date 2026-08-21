"""Abstract base class for systolic-array backends.

See `systolic` for a general overview.
"""

import abc

from torch import Tensor

from systolic._rust import Fault


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
