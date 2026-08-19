"""Systolic-array backend abstraction and layer mapping.

`SystolicBackend` is a weight-stationary systolic-array evaluation strategy
(not a generic matmul backend): `matmul` takes `weights (out_features,
in_features)` and `activations (in_features, batch)` and returns
`(out_features, batch)`, matching the physical roles the later fault-lifting
recipes depend on. It also carries array geometry (`nrows`/`ncols`) and fault
state (`set_fault`); `TorchBackend`, the always-correct reference
implementation, has no array constraints or notion of faults and raises if
those are used.

`MappedLinear`/`MappedConv2d` replace `nn.Linear`/`nn.Conv2d` with versions
that route their matmul through a `SystolicBackend`, bridging PyTorch's
row-vector convention to the backend's hardware convention at exactly this
boundary. `BackendModel` recursively performs that replacement over an
existing `nn.Module`.

`SimulatedBackend` is the cycle-accurate oracle, `LiftedBackend` the fast
torch-side workhorse (Rust computes the lift, torch applies it). Both run
register faults (`Fault.Register`) restricted to a `RegisterFaults` subset,
in float32 only.
"""

from faultforge._internal.systolic.backend import SystolicBackend
from faultforge._internal.systolic.fault import RegisterFaults
from faultforge._internal.systolic.layers import MappedConv2d, MappedLinear
from faultforge._internal.systolic.lifted_backend import LiftedBackend
from faultforge._internal.systolic.model import BackendModel
from faultforge._internal.systolic.simulated_backend import SimulatedBackend
from faultforge._internal.systolic.torch_backend import TorchBackend
from faultforge._rust.systolic import (
    AccumulatorFaultPart,
    ArrayConfig,
    Fault,
    Index2,
    LiftedFault,
    PeRegisterKind,
    StuckAtKind,
)

__all__ = [
    "AccumulatorFaultPart",
    "ArrayConfig",
    "BackendModel",
    "Fault",
    "Index2",
    "LiftedBackend",
    "LiftedFault",
    "MappedConv2d",
    "MappedLinear",
    "PeRegisterKind",
    "RegisterFaults",
    "SimulatedBackend",
    "StuckAtKind",
    "SystolicBackend",
    "TorchBackend",
]
