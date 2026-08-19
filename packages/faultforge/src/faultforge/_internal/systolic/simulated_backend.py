"""The cycle-accurate systolic-array oracle backend."""

from typing import final, override

import torch
from torch import Tensor

from faultforge._internal.systolic.backend import SystolicBackend
from faultforge._rust.systolic import Fault, Mapping, simulated_matmul


@final
class SimulatedBackend(SystolicBackend):
    """Runs the literal, cycle-accurate systolic array simulation.

    Correct by construction (it's the same run loop the physical array would
    execute), but simulates the array cycle by cycle on the CPU, so it's much
    slower than `LiftedBackend`. Meant for validation and cross-checks, not for
    large campaigns. f32 only.
    """

    def __init__(self, nrows: int, ncols: int) -> None:
        self._nrows = nrows
        self._ncols = ncols
        self._fault: Fault | None = None
        # One Mapping per distinct weight shape seen so far: a physical
        # fault lifts/simulates differently per layer, but the mapping
        # itself only depends on the weight shape and array size.
        self._mappings: dict[torch.Size, Mapping] = {}

    def _mapping_for(self, weights: Tensor) -> Mapping:
        mapping = self._mappings.get(weights.shape)
        if mapping is None:
            mapping = Mapping.auto_for(
                weights.numpy(force=True), self._nrows, self._ncols
            )
            self._mappings[weights.shape] = mapping
        return mapping

    @override
    def matmul(self, weights: Tensor, activations: Tensor) -> Tensor:
        if weights.dtype != torch.float32:
            raise ValueError(
                f"SimulatedBackend only supports float32, got {weights.dtype}"
            )

        mapping = self._mapping_for(weights)
        result = simulated_matmul(
            mapping,
            weights.numpy(force=True),
            activations.numpy(force=True),
            self._nrows,
            self._ncols,
            self._fault,
        )
        return torch.from_numpy(result)

    @override
    def set_fault(self, fault: Fault | None) -> None:
        self._fault = fault

    @override
    def nrows(self) -> int:
        return self._nrows

    @override
    def ncols(self) -> int:
        return self._ncols
