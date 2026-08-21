"""Abstract base classes for encoders and encodings.

See `faultforge.encoding` for a general overview.
"""

import abc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from faultforge._internal.dtype import EncodingDtype
from faultforge._internal.fault import Fault
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.progress import Progress, stage
from faultforge._internal.tensor import (
    tensor_list_dtype,
    tensor_list_fault,
    tensor_list_faults,
)

logger = logging.getLogger(__name__)


class Encoder(abc.ABC):
    """An encoder for lists of tensors."""

    @abc.abstractmethod
    def encode(self, ts: list[Tensor], *, progress: Progress | None = None) -> Encoding:
        """Encode a list of tensors."""
        ...

    @abc.abstractmethod
    def fingerprint(self) -> Fingerprint:
        """Return a structural identity for this encoder.

        Two encoders that encode tensors identically should produce equal
        fingerprints. Composite encoders nest the fingerprints of their parts.
        """
        ...


class Encoding:
    """An encoded list of tensors.

    Created by an `Encoder`.
    """

    @abc.abstractmethod
    def decode(self) -> list[Tensor]:
        """Decode to a list of tensors."""
        ...

    @abc.abstractmethod
    def clone(self) -> Encoding:
        """Clone the encoded data."""
        ...

    @abc.abstractmethod
    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        """Apply a fault to the encoded data at the given bit index."""
        ...

    def apply_faults(self, faults: Sequence[tuple[Fault, int]]) -> None:
        """Apply multiple faults at once.

        The default implementation just calls `apply_fault` in a loop.
        Encodings that can apply a batch more efficiently than one fault at a
        time (e.g. by only paying a per-call conversion overhead once for the
        whole batch) should override this.
        """
        for fault, target_bit in faults:
            self.apply_fault(fault, target_bit)

    @abc.abstractmethod
    def bit_count(self) -> int:
        """Return the number of bits in the encoded data."""
        ...


class TensorEncoder(Encoder):
    """An Encoder which produces a `TensorEncoding`."""

    @override
    @abc.abstractmethod
    def encode(
        self, ts: list[Tensor], *, progress: Progress | None = None
    ) -> TensorEncoding: ...


class TensorEncoding(Encoding):
    """An encoding which uses tensors as the underlying data structure.

    These kinds of encodings can be sequenced.
    """

    @abc.abstractmethod
    def encoded_tensors(self) -> list[Tensor]:
        """Access the encoded tensors.

        Make sure to call `trigger_recompute` when modifying the encoded
        tensors.
        """
        ...

    @abc.abstractmethod
    def trigger_recompute(self) -> None:
        """Trigger a recompute of the encoded tensors.

        This should be called when the encoded tensors have been modified.
        """
        ...

    @override
    @abc.abstractmethod
    def clone(self) -> TensorEncoding:
        """Clone the encoding."""
        ...


class InPlaceEncoder(TensorEncoder):
    """A helper class for implementing `TensorEncoder`."""

    @abc.abstractmethod
    def encode_float32(self, t: Tensor) -> Tensor:
        """Encode a float32 tensor."""
        ...

    @abc.abstractmethod
    def encode_float16(self, t: Tensor) -> Tensor:
        """Encode a float16 tensor."""
        ...

    @abc.abstractmethod
    def create_encoding(
        self,
        data: list[Tensor],
        bit_count: int,
        dtype: EncodingDtype,
    ) -> TensorEncoding:
        """Create the concrete encoding instance."""
        ...

    @override
    def encode(
        self, ts: list[Tensor], *, progress: Progress | None = None
    ) -> TensorEncoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty list")

        dtype = EncodingDtype.from_torch(dtype)

        bit_count = 0
        data: list[Tensor] = []
        element_bit_count = dtype.bit_count()

        match dtype:
            case EncodingDtype.F32:
                encode = self.encode_float32
            case EncodingDtype.F16:
                encode = self.encode_float16

        with stage(
            progress, f"Encoding ({self.fingerprint().kind})", total=len(ts)
        ) as s:
            for t in ts:
                bit_count += t.numel() * element_bit_count
                data.append(encode(t))
                s.advance()

        return self.create_encoding(data, bit_count, dtype)


@dataclass
class InPlaceEncoding(TensorEncoding):
    """A helper base class for implementing TensorEncoding"""

    _encoded_data: list[Tensor]
    _bit_count: int
    _decoded_tensors: list[Tensor] | None
    _dtype: EncodingDtype

    @abc.abstractmethod
    def decode_float32(self, t: Tensor) -> Tensor: ...

    @abc.abstractmethod
    def decode_float16(self, t: Tensor) -> Tensor: ...

    @override
    def encoded_tensors(self) -> list[Tensor]:
        return self._encoded_data

    @override
    def trigger_recompute(self) -> None:
        self._decoded_tensors = None

    @override
    def decode(self) -> list[Tensor]:
        if self._decoded_tensors is not None:
            return self._decoded_tensors

        match self._dtype:
            case EncodingDtype.F16:
                decode = self.decode_float16
            case EncodingDtype.F32:
                decode = self.decode_float32

        decoded = []
        for encoded in self.encoded_tensors():
            with torch.no_grad():
                decoded_item = decode(encoded)
            decoded.append(decoded_item)

        self._decoded_tensors = decoded
        return decoded

    def _invalidate_decoded_cache(self) -> None:
        if self._decoded_tensors is not None:
            logger.debug("Invalidating decoded tensors due to fault injection")
        self._decoded_tensors = None

    @override
    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        self._invalidate_decoded_cache()
        tensor_list_fault(self._encoded_data, fault, target_bit)

    @override
    def apply_faults(self, faults: Sequence[tuple[Fault, int]]) -> None:
        self._invalidate_decoded_cache()
        tensor_list_faults(self._encoded_data, list(faults))

    @override
    def bit_count(self) -> int:
        return self._bit_count
