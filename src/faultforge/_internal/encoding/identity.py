"""An encoder that stores tensors unmodified, without any protection."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from torch import Tensor

from faultforge._internal.dtype import EncodingDtype
from faultforge._internal.encoding.abc import TensorEncoder, TensorEncoding
from faultforge._internal.fault import Fault
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.progress import Progress
from faultforge._internal.tensor import (
    tensor_list_dtype,
    tensor_list_fault,
    tensor_list_faults,
)


class IdentityEncoder(TensorEncoder):
    """An encoder for `IdentityEncoding`."""

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="identity")

    @override
    def encode(
        self, ts: list[Tensor], *, progress: Progress | None = None
    ) -> IdentityEncoding:
        _ = progress
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty list")
        dtype = EncodingDtype.from_torch(dtype)
        bit_count = sum(t.numel() for t in ts) * dtype.bit_count()
        return IdentityEncoding(_tensors=list(ts), _bit_count=bit_count)


@dataclass
class IdentityEncoding(TensorEncoding):
    """An encoding that stores tensors unmodified, without any protection.

    This can be used in place of an actual encoder for functions which expect
    it. Useful as a baseline to compare protected encodings against.
    """

    _tensors: list[Tensor]
    _bit_count: int

    @override
    def encoded_tensors(self) -> list[Tensor]:
        return self._tensors

    @override
    def trigger_recompute(self) -> None:
        pass

    @override
    def decode(self) -> list[Tensor]:
        return self._tensors

    @override
    def clone(self) -> IdentityEncoding:
        return IdentityEncoding(
            _tensors=[t.clone() for t in self._tensors], _bit_count=self._bit_count
        )

    @override
    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        tensor_list_fault(self._tensors, fault, target_bit)

    @override
    def apply_faults(self, faults: Sequence[tuple[Fault, int]]) -> None:
        tensor_list_faults(self._tensors, list(faults))

    @override
    def bit_count(self) -> int:
        return self._bit_count
