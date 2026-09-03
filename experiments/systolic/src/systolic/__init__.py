"""An experiment for measuring model reliability under systolic-array
register faults.

`SystolicFaultInjection` runs a model whose `nn.Linear`/`nn.Conv2d` layers
are routed through a `SystolicBackend`, injects a stuck-at register fault
into one processing element, and scores the result according to a
`ReliabilityMetric`.

`SystolicBackend` is a weight-stationary systolic-array evaluation strategy
(not a generic matmul backend): `matmul` takes `weights (out_features,
in_features)` and `activations (in_features, batch)` and returns
`(out_features, batch)`, matching the physical roles the fault-lifting
recipes depend on. `TorchBackend` is the always-correct, fault-free
reference implementation; `SimulatedBackend` is the cycle-accurate oracle;
`LiftedBackend` is the fast torch-side workhorse (Rust computes the lift,
torch applies it). `MappedLinear`/`MappedConv2d` replace `nn.Linear`/
`nn.Conv2d` with versions that route their matmul through a
`SystolicBackend`; `BackendModel` recursively performs that replacement over
an existing `nn.Module`.

`run_profiling` walks a model's mapped layers over a random dataset
subsample through a `ProfilingBackend`, pooling every layer's and batch's
MAC inputs into one `ProfilingArtifact` - the per-PE input distribution
later fault-modeling work samples from. `save_profiling_artifact`/
`load_profiling_artifact` round-trip that artifact to a single `.npz` file.

`gap_grids` compares each PE's profiled inputs against the pooled array (see
`GapGrid`), a first, cheap look at whether faults need modeling per row,
column, or PE rather than for the array as a whole. `build_gap_heatmap_figure`
draws the result.
"""

from systolic._rust import (
    AccumulatorFaultPart,
    ArrayConfig,
    Fault,
    Index2,
    LiftedFault,
    Mapping,
    Pass,
    PeRegisterKind,
    StuckAtKind,
)
from systolic.backend import SystolicBackend
from systolic.experiment import (
    BackendKind,
    ReliabilityMetric,
    SavedResult,
    SystolicFaultInjection,
)
from systolic.fault import RegisterFaults
from systolic.layers import MappedConv2d, MappedLinear
from systolic.lifted_backend import LiftedBackend
from systolic.model import BackendModel
from systolic.profiling import (
    ProfilingMetadata,
    load_profiling_artifact,
    save_profiling_artifact,
)
from systolic.profiling_backend import ProfilingBackend
from systolic.profiling_driver import run_profiling
from systolic.profiling_plots import build_gap_heatmap_figure
from systolic.profiling_similarity import GapGrid, Regime, Variable, gap_grids
from systolic.simulated_backend import SimulatedBackend
from systolic.torch_backend import TorchBackend

__all__ = [
    "AccumulatorFaultPart",
    "ArrayConfig",
    "BackendKind",
    "BackendModel",
    "Fault",
    "GapGrid",
    "Index2",
    "LiftedBackend",
    "LiftedFault",
    "Mapping",
    "MappedConv2d",
    "MappedLinear",
    "Pass",
    "PeRegisterKind",
    "ProfilingBackend",
    "ProfilingMetadata",
    "Regime",
    "RegisterFaults",
    "ReliabilityMetric",
    "SavedResult",
    "SimulatedBackend",
    "StuckAtKind",
    "SystolicBackend",
    "SystolicFaultInjection",
    "TorchBackend",
    "Variable",
    "build_gap_heatmap_figure",
    "gap_grids",
    "load_profiling_artifact",
    "run_profiling",
    "save_profiling_artifact",
]
