"""Torch port of `LiftedRegisterFault::matmul`
(`crates/systolic/src/fault/register_lift.rs`), arm for arm. See
`docs/fault-lifting.md` for the theory behind each recipe.
"""

import torch
from torch import Tensor

from faultforge._rust.systolic import Fault, LiftedFault, StuckAtKind


def _corrupt(tensor: Tensor, stuck_at: StuckAtKind, bit_index: int) -> Tensor:
    """Force `bit_index` of every element to `stuck_at`, via a same-width int view."""
    if tensor.dtype != torch.float32:
        raise ValueError(f"corrupt() only supports float32, got {tensor.dtype}")

    # An unsigned view means `1 << bit_index` (up to bit 31) is always
    # representable directly, unlike a signed int32 view where it would
    # overflow the sign bit.
    view = tensor.view(torch.uint32)
    mask = 1 << bit_index
    match stuck_at:
        case StuckAtKind.One:
            corrupted = view | mask
        case StuckAtKind.Zero:
            corrupted = view & ~mask
    return corrupted.view(tensor.dtype)


def _weight_fault_matmul(
    weights: Tensor,
    activations: Tensor,
    fault: Fault.Register,
    lifted: LiftedFault.Weight,
) -> Tensor:
    weights = weights.clone()
    for index in lifted.affected_weights:
        weights[index.y, index.x] = _corrupt(
            weights[index.y, index.x], fault.stuck_at, fault.bit_index
        )
    return weights @ activations


def _activation_fault_matmul(
    weights: Tensor,
    activations: Tensor,
    fault: Fault.Register,
    lifted: LiftedFault.Activation,
) -> Tensor:
    output = weights @ activations

    activations_faulty = activations.clone()
    for row_index in lifted.affected_activation_rows:
        activations_faulty[row_index, :] = _corrupt(
            activations_faulty[row_index, :], fault.stuck_at, fault.bit_index
        )
    output_faulty = weights @ activations_faulty

    for row_index in lifted.affected_output_rows:
        output[row_index, :] = output_faulty[row_index, :]

    return output


def _accumulator_fault_matmul(
    weights: Tensor,
    activations: Tensor,
    fault: Fault.Register,
    lifted: LiftedFault.Accumulator,
) -> Tensor:
    output = weights @ activations

    for part in lifted.parts:
        start, end = part.for_activations
        row = part.affected_output_row

        # An empty range means no products had reached the faulty PE yet
        # (a fault above the pass's used band): the partial sum is zero, a
        # "faulty zero" rather than a separate case. See
        # `docs/fault-lifting.md`'s accumulator section.
        if start == end:
            partial_sums = torch.zeros(
                activations.shape[1], dtype=activations.dtype, device=activations.device
            )
        else:
            partial_sum_weights = weights[row, start:end]
            partial_sum_activations = activations[start:end, :]
            partial_sums = partial_sum_weights @ partial_sum_activations

        corrupted = _corrupt(partial_sums, fault.stuck_at, fault.bit_index)
        output[row, :] = output[row, :] - partial_sums + corrupted

    return output


def apply_lifted_register_fault(
    lifted: LiftedFault, fault: Fault, weights: Tensor, activations: Tensor
) -> Tensor:
    """Apply a lifted register fault to a matmul.

    Produces a result equivalent to running the same fault through the
    literal systolic array (`SimulatedBackend`), without simulating the
    array.
    """
    if not isinstance(fault, Fault.Register):
        raise TypeError(f"unsupported fault kind: {type(fault)!r}")

    if isinstance(lifted, LiftedFault.Weight):
        return _weight_fault_matmul(weights, activations, fault, lifted)
    elif isinstance(lifted, LiftedFault.Activation):
        return _activation_fault_matmul(weights, activations, fault, lifted)
    elif isinstance(lifted, LiftedFault.Accumulator):
        return _accumulator_fault_matmul(weights, activations, fault, lifted)
    else:
        raise TypeError(f"unsupported lifted fault kind: {type(lifted)!r}")
