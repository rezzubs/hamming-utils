"""Recursively map a model's layers onto a SystolicBackend."""

from typing import Any, final, override

from faultforge.progress import Progress, stage
from torch import nn

from systolic.backend import SystolicBackend
from systolic.layers import MappedConv2d, MappedLinear


def _qualified_name(prefix: str, name: str) -> str:
    """Join a dotted path prefix with a child's `named_children` name.

    `_wrapped` is `BackendModel`'s own attribute name for the model it wraps,
    an implementation detail rather than part of the wrapped model's naming,
    so it's dropped rather than prefixed onto every layer's label.
    """
    if name == "_wrapped" and not prefix:
        return ""
    return f"{prefix}.{name}" if prefix else name


def _count_mappable(layer: nn.Module) -> int:
    """How many of `layer`'s submodules (recursively) can be mapped.

    Counted up front so each mapped layer can be labelled with its position
    among them.
    """
    count = 0
    for child in layer.children():
        if isinstance(child, nn.Linear | nn.Conv2d):
            count += 1
        else:
            count += _count_mappable(child)
    return count


def _map_recursive(
    layer: nn.Module,
    backend: SystolicBackend,
    progress: Progress | None,
    prefix: str,
    next_index: int,
    total: int,
) -> int:
    """Replace every supported submodule of `layer` (recursively) with a
    mapped equivalent, returning the first position number it did not use.

    `next_index` and `total` label each mapped layer with its position in the
    module tree, so a progress line names where in the model a layer sits.
    """
    for name, child in layer.named_children():
        qualified_name = _qualified_name(prefix, name)
        position = f"{next_index}/{total}"
        if isinstance(child, nn.Linear):
            setattr(
                layer,
                name,
                MappedLinear(
                    child,
                    backend,
                    name=qualified_name,
                    position=position,
                    progress=progress,
                ),
            )
            next_index += 1
        elif isinstance(child, nn.Conv2d):
            setattr(
                layer,
                name,
                MappedConv2d(
                    child,
                    backend,
                    name=qualified_name,
                    position=position,
                    progress=progress,
                ),
            )
            next_index += 1
        else:
            next_index = _map_recursive(
                child, backend, progress, qualified_name, next_index, total
            )
        # Reassigning an existing child name is safe during iteration (dict
        # value replacement, not insertion/deletion), so this doesn't fight
        # Python's "can't mutate size during iteration" rule.
    return next_index


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

    def __init__(
        self,
        wrapped: nn.Module,
        backend: SystolicBackend,
        *,
        progress: Progress | None = None,
    ) -> None:
        super().__init__()
        self._wrapped = wrapped
        self._progress = progress
        _ = _map_recursive(self, backend, progress, "", 1, _count_mappable(self))

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # No `total`: the wrapped model's own forward drives the mapped layers,
        # so nothing here is in a position to count them off. Each layer
        # instead carries its position in its own label.
        with stage(self._progress, "forward"):
            return self._wrapped.forward(*args, **kwargs)
