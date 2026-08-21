"""Operations on tensors."""

import numpy as np
import torch
from torch import Tensor

from faultforge import _rust
from faultforge._internal.dtype import FiDtype
from faultforge._internal.fault import (
    Fault,
    fault_to_rust,
)


def bitwise_xor(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise bitwise xor of two tensors.

    Floats are bitcast to same-width integers first, since `torch.bitwise_xor`
    only supports integer/bool dtypes.

    Raises:
        ValueError: If the data type is unsupported. See `FiDtype`.
    """
    assert a.shape == b.shape
    assert a.dtype == b.dtype

    match FiDtype.from_torch(a.dtype):
        case FiDtype.F32:
            return torch.bitwise_xor(a.view(torch.int32), b.view(torch.int32))
        case FiDtype.F16:
            return torch.bitwise_xor(a.view(torch.int16), b.view(torch.int16))
        case FiDtype.U8:
            return torch.bitwise_xor(a, b)


def tensor_list_dtype(ts: list[torch.Tensor]) -> torch.dtype | None:
    """Confirms that all tensors in `ts` have the same datatype.

    Returns:
        The common dtype of `ts` or None if `ts` is empty.

    Raises:
        ValueError: If dtype values don't match.
    """

    dtype = None

    for i, t in enumerate(ts):
        if dtype is None:
            dtype = t.dtype
        else:
            if dtype != t.dtype:
                raise ValueError(
                    f"dtype=`{t.dtype}` for tensor {i} while all previous values had dtype=`{dtype}`"
                )
    return dtype


def tensor_list_fault(ts: list[torch.Tensor], fault: Fault, target_bit: int):
    """Apply a fault at specific bit position.

    Raises:
        ValueError:
            - If values in `ts` don't all have the same data type.
            - If the data type is unsupported. See `FiDtype`.
    """
    tensor_list_faults(ts, [(fault, target_bit)])


def tensor_list_faults(ts: list[torch.Tensor], faults: list[tuple[Fault, int]]) -> None:
    """Apply multiple faults, given as `(fault, target_bit)` pairs.

    Every tensor in `ts` is converted to numpy and copied back exactly once
    for the whole batch rather than once per fault, which matters a lot: a
    single tensor's `numpy(force=True)`/`copy_()` round-trip is O(numel), so
    doing it per-fault instead of per-batch turns fault injection from
    O(faults * numel) into O(faults + numel).

    Raises:
        ValueError:
            - If values in `ts` don't all have the same data type.
            - If the data type is unsupported. See `FiDtype`.
    """

    dtype = tensor_list_dtype(ts)

    if dtype is None:
        raise ValueError("`ts` is empty")

    rust_faults = [(fault_to_rust(fault), target_bit) for fault, target_bit in faults]

    # NOTE: the length checks are handled in rust.
    match FiDtype.from_torch(dtype):
        case FiDtype.F32:
            with torch.no_grad():
                np_array = [t.numpy(force=True) for t in ts]
                _rust.list_of_array_faults_f32(np_array, rust_faults)

        case FiDtype.F16:
            with torch.no_grad():
                np_array = [t.numpy(force=True).view(np.uint16) for t in ts]
                _rust.list_of_array_faults_u16(np_array, rust_faults)
                np_array = [t.view(np.float16) for t in np_array]
        case FiDtype.U8:
            with torch.no_grad():
                np_array = [t.numpy(force=True) for t in ts]
                _rust.list_of_array_faults_u8(np_array, rust_faults)

    for original, updated in zip(ts, np_array, strict=True):
        # We have to assert because Tensor.numpy returns `Unknown`.
        assert isinstance(updated, np.ndarray)
        updated = torch.from_numpy(updated)
        assert updated.shape == original.shape

        with torch.no_grad():
            _ = original.copy_(updated)
