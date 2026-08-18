"""Systolic-array backend abstraction and layer mapping.

`SystolicBackend` is a weight-stationary systolic-array evaluation strategy
(not a generic matmul backend): `matmul` takes `weights (out_features,
in_features)` and `activations (in_features, batch)` and returns
`(out_features, batch)`, matching the physical roles the later fault-lifting
recipes depend on. `TorchBackend` is the always-correct reference
implementation.

`MappedLinear`/`MappedConv2d` replace `nn.Linear`/`nn.Conv2d` with versions
that route their matmul through a `SystolicBackend`, bridging PyTorch's
row-vector convention to the backend's hardware convention at exactly this
boundary. `BackendModel` recursively performs that replacement over an
existing `nn.Module`.
"""

from faultforge._internal.systolic.backend import SystolicBackend
from faultforge._internal.systolic.layers import MappedConv2d, MappedLinear
from faultforge._internal.systolic.model import BackendModel
from faultforge._internal.systolic.torch_backend import TorchBackend

__all__ = [
    "BackendModel",
    "MappedConv2d",
    "MappedLinear",
    "SystolicBackend",
    "TorchBackend",
]
