"""The torch-side workhorse backend: Rust lifts, Torch applies."""

from typing import final, override

import torch
from torch import Tensor

from systolic._rust import Fault, LiftedFault
from systolic.backend import MappingCache, SystolicBackend
from systolic.lift_apply import apply_lifted_register_fault


@final
class LiftedBackend(SystolicBackend):
    """Applies register faults via Rust-computed lift descriptions.

    A lift computes which values of the input/output matrices would be affected
    by the faulty systolic array operation and executes these effects using
    torch operations instead of running a cycle-accurate systolic array
    simulator (like `SimulatedBackend`).
    """

    def __init__(self, nrows: int, ncols: int) -> None:
        self._nrows = nrows
        self._ncols = ncols
        self._fault: Fault | None = None
        self._mapping_cache = MappingCache(nrows, ncols)
        # A single physical fault lifts differently per weight shape, so the
        # lift result is cached per (shape, fault) and reused across every
        # batch of a run.
        self._lifts: dict[tuple[torch.Size, Fault], LiftedFault] = {}

    @override
    def matmul(self, weights: Tensor, activations: Tensor) -> Tensor:
        if weights.dtype != torch.float32:
            raise ValueError(
                f"LiftedBackend only supports float32, got {weights.dtype}"
            )

        if self._fault is None:
            return weights @ activations

        mapping = self._mapping_cache.get(weights)
        key = (weights.shape, self._fault)
        lifted = self._lifts.get(key)
        if lifted is None:
            lifted = mapping.lift(self._fault)
            self._lifts[key] = lifted

        return apply_lifted_register_fault(lifted, self._fault, weights, activations)

    @override
    def set_fault(self, fault: Fault | None) -> None:
        self._fault = fault

    @override
    def nrows(self) -> int:
        return self._nrows

    @override
    def ncols(self) -> int:
        return self._ncols
