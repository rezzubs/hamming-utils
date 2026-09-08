"""The golden reference systolic backend."""

from typing import final, override

from torch import Tensor

from systolic._rust import Fault
from systolic.backend import SystolicBackend


@final
class TorchBackend(SystolicBackend):
    """Computes `weights @ activations` directly via PyTorch.

    Has no array constraints and no notion of faults. Used as the
    fault-free golden path.
    """

    @override
    def matmul(self, weights: Tensor, activations: Tensor) -> Tensor:
        return weights @ activations

    @override
    def set_fault(self, fault: Fault | None) -> None:
        raise NotImplementedError("TorchBackend has no notion of faults")

    @override
    def nrows(self) -> int:
        raise NotImplementedError("TorchBackend has no array constraints")

    @override
    def ncols(self) -> int:
        raise NotImplementedError("TorchBackend has no array constraints")
