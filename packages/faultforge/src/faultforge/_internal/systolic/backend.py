"""Abstract base class for systolic-array backends.

See `faultforge.systolic` for a general overview.
"""

import abc

from torch import Tensor


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
