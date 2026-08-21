"""Recursively map a model's layers onto a SystolicBackend."""

from typing import Any, final, override

from torch import nn

from systolic.backend import SystolicBackend
from systolic.layers import MappedConv2d, MappedLinear


def _map_recursive(layer: nn.Module, backend: SystolicBackend) -> None:
    for name, child in layer.named_children():
        if isinstance(child, nn.Linear):
            setattr(layer, name, MappedLinear(child, backend))
        elif isinstance(child, nn.Conv2d):
            setattr(layer, name, MappedConv2d(child, backend))
        else:
            _map_recursive(child, backend)
        # Reassigning an existing child name is safe during iteration (dict
        # value replacement, not insertion/deletion), so this doesn't fight
        # Python's "can't mutate size during iteration" rule.


@final
class BackendModel(nn.Module):
    """Wraps `wrapped`, replacing every supported submodule
    (recursively) with one that routes its matmul through `backend`.

    Currently supported Module types:
    - torch.nn.Conv2d (only group=1)
    - torch.nn.Linear

    Mutates `wrapped`'s submodule tree in place. `wrapped` becomes a registered
    child of this module, and replacement happens by `setattr`ing into
    it directly instead of copying. If you need an untouched reference model
    (e.g. for a golden baseline), `copy.deepcopy` it before passing it here;
    `BackendModel` itself never copies.
    """

    def __init__(self, wrapped: nn.Module, backend: SystolicBackend) -> None:
        super().__init__()
        self._wrapped = wrapped
        _map_recursive(self, backend)

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._wrapped.forward(*args, **kwargs)
