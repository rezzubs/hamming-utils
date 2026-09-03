"""The cycle-accurate systolic-array oracle backend."""

from typing import final, override

import torch
from torch import Tensor

from systolic._rust import Fault, simulated_matmul
from systolic.backend import MappingCache, SystolicBackend


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
        self._mapping_cache = MappingCache(nrows, ncols)

    @override
    def matmul(self, weights: Tensor, activations: Tensor) -> Tensor:
        if weights.dtype != torch.float32:
            raise ValueError(
                f"SimulatedBackend only supports float32, got {weights.dtype}"
            )

        mapping = self._mapping_cache.get(weights)
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
