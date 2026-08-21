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
from systolic.simulated_backend import SimulatedBackend
from systolic.torch_backend import TorchBackend

__all__ = [
    "AccumulatorFaultPart",
    "ArrayConfig",
    "BackendKind",
    "BackendModel",
    "Fault",
    "Index2",
    "LiftedBackend",
    "LiftedFault",
    "Mapping",
    "MappedConv2d",
    "MappedLinear",
    "Pass",
    "PeRegisterKind",
    "RegisterFaults",
    "ReliabilityMetric",
    "SavedResult",
    "SimulatedBackend",
    "StuckAtKind",
    "SystolicBackend",
    "SystolicFaultInjection",
    "TorchBackend",
]
