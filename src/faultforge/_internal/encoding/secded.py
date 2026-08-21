"""Single Error Correction Double Error Detection (SECDED) encoding."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

import torch

from faultforge._internal.dtype import EncodingDtype
from faultforge._internal.encoding.abc import Encoder, Encoding
from faultforge._internal.fault import (
    Fault,
    fault_to_rust,
)
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.progress import Progress, stage
from faultforge._internal.tensor import tensor_list_dtype
from faultforge._rust import secded

logger = logging.getLogger(__name__)


@dataclass
class SecdedEncoder(Encoder):
    """Encodes tensors with Single Error Correction Double Error Detection (SECDED) hamming codes.

    Each chunk of `bits_per_chunk` data bits gets its own hamming code, which
    allows single-bit errors within the chunk to be corrected and double-bit
    errors to be detected during decoding. Double-error detection results are
    not currently used.
    """

    bits_per_chunk: int
    """The number of data bits protected by a single hamming code.

    Equivalent to the memory line size in hardware. Can be any positive
    integer but multiples of 8 bits have better encoding/decoding performance.
    """

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(
            kind="secded", scalars={"bits_per_chunk": self.bits_per_chunk}
        )

    @override
    def encode(
        self, ts: list[torch.Tensor], *, progress: Progress | None = None
    ) -> Encoding:
        dtype = tensor_list_dtype(ts)
        if dtype is None:
            raise ValueError("Cannot encode an empty buffer")
        dtype = EncodingDtype.from_torch(dtype)

        # Store decoded tensor copies
        decoded_tensors = [t.clone() for t in ts]

        with stage(progress, "Encoding (SECDED)"):
            match dtype:
                case EncodingDtype.F32:
                    with torch.no_grad():
                        rust_input = [t.flatten().numpy(force=True) for t in ts]
                    encoded_data = secded.encode_f32(rust_input, self.bits_per_chunk)
                case EncodingDtype.F16:
                    with torch.no_grad():
                        rust_input = [
                            t.flatten().view(torch.uint16).numpy(force=True) for t in ts
                        ]
                    encoded_data = secded.encode_u16(rust_input, self.bits_per_chunk)

        return SecdedEncoding(
            encoded_data,
            decoded_tensors,
            dtype,
        )


@dataclass
class SecdedEncoding(Encoding):
    """The encoding produced by `SecdedEncoder`. See `SecdedEncoder` for details."""

    _encoded_data: secded.Encoding
    """The blob that stores raw encoded data."""
    _decoded_tensors: list[torch.Tensor]
    """These are updated in-place during decoding."""
    _dtype: EncodingDtype
    _needs_recompute: bool = False

    @override
    def decode(self) -> list[torch.Tensor]:
        if not self._needs_recompute:
            logger.debug("Using cached decoded tensors")
            return self._decoded_tensors
        logger.debug("Recomputing decoded tensors")

        match self._dtype:
            case EncodingDtype.F32:
                decoded, ded_results = self._encoded_data.decode_f32()
                torch_decoded = [torch.from_numpy(t) for t in decoded]

            case EncodingDtype.F16:
                decoded, ded_results = self._encoded_data.decode_u16()
                torch_decoded = [
                    torch.from_numpy(t).view(torch.float16) for t in decoded
                ]

        # Update cached decoded tensors in-place
        for cached, decoded in zip(self._decoded_tensors, torch_decoded, strict=True):
            with torch.no_grad():
                _ = cached.flatten().copy_(decoded)

        # We're discarding the double error detection results for now but may
        # want to do something with them in the future.
        _ = ded_results

        self._needs_recompute = False
        return self._decoded_tensors

    @override
    def clone(self) -> SecdedEncoding:
        return SecdedEncoding(
            self._encoded_data.clone(),
            [t.clone() for t in self._decoded_tensors],
            self._dtype,
            self._needs_recompute,
        )

    def _invalidate_decoded_cache(self) -> None:
        if not self._needs_recompute:
            logger.debug("Invalidating decoded tensor cache due to fault injection.")
        self._needs_recompute = True

    @override
    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        self._invalidate_decoded_cache()
        self._encoded_data.apply_fault(fault_to_rust(fault), target_bit)

    @override
    def apply_faults(self, faults: Sequence[tuple[Fault, int]]) -> None:
        self._invalidate_decoded_cache()
        self._encoded_data.apply_faults(
            [(fault_to_rust(fault), target_bit) for fault, target_bit in faults]
        )

    @override
    def bit_count(self) -> int:
        return self._encoded_data.bit_count()
