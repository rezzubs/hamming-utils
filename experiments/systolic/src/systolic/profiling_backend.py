"""A backend that records per-PE logic inputs instead of faulting anything."""

from typing import final, override

import torch
from torch import Tensor

from systolic._rust import Fault, Profiler, ProfilingArtifact
from systolic.backend import MappingCache, SystolicBackend


@final
class ProfilingBackend(SystolicBackend):
    """Routes matmuls through a `Profiler`, pooling every call - across
    every wrapped layer - into one profiling artifact.

    Has no notion of faults: profiling records the fault-free MAC inputs a
    model actually sees.
    """

    def __init__(self, nrows: int, ncols: int, capacity: int, seed: int) -> None:
        self._nrows = nrows
        self._ncols = ncols
        self._profiler = Profiler(nrows, ncols, capacity, seed)
        self._mapping_cache = MappingCache(nrows, ncols)

    @override
    def matmul(self, weights: Tensor, activations: Tensor) -> Tensor:
        if weights.dtype != torch.float32:
            raise ValueError(
                f"ProfilingBackend only supports float32, got {weights.dtype}"
            )

        mapping = self._mapping_cache.get(weights)
        result = self._profiler.run(
            mapping, weights.numpy(force=True), activations.numpy(force=True)
        )
        return torch.from_numpy(result)

    @override
    def set_fault(self, fault: Fault | None) -> None:
        raise NotImplementedError("ProfilingBackend has no notion of faults")

    @override
    def nrows(self) -> int:
        return self._nrows

    @override
    def ncols(self) -> int:
        return self._ncols

    def artifact(self) -> ProfilingArtifact:
        """Snapshot everything accumulated so far.

        Non-destructive - callable more than once, e.g. to checkpoint
        mid-run. See `Profiler.artifact`.
        """
        return self._profiler.artifact()
