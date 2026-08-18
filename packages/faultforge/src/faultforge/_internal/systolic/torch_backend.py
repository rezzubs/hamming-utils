"""The golden reference systolic backend."""

from typing import final, override

from torch import Tensor

from faultforge._internal.systolic.backend import SystolicBackend


@final
class TorchBackend(SystolicBackend):
    """Computes `weights @ activations` directly via PyTorch.

    Has no array constraints and no notion of faults. Used as the
    fault-free golden path.
    """

    @override
    def matmul(self, weights: Tensor, activations: Tensor) -> Tensor:
        return weights @ activations
