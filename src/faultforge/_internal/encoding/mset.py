"""Most Significant Exponent bit Triplication (MSET)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import final, override

import torch
from torch import Tensor

from faultforge._internal.dtype import EncodingDtype
from faultforge._internal.encoding.abc import (
    InPlaceEncoder,
    InPlaceEncoding,
    TensorEncoding,
)
from faultforge._internal.fingerprint import Fingerprint
from faultforge._rust import mset

_logger = logging.getLogger(__name__)


@final
class MsetEncoder(InPlaceEncoder):
    """Encodes tensors with Most Significant Exponent bit Triplication (MSET).

    The second highest bit of each element is copied to the two lowest bits.
    A majority voting scheme determines the final value of the exponent bit
    during decoding.
    """

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="mset")

    @override
    def encode_float32(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.numpy(force=True)
        mset.encode_f32(t_np)
        return torch.from_numpy(t_np)

    @override
    def encode_float16(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.view(torch.uint16).numpy(force=True)
        mset.encode_u16(t_np)
        return torch.from_numpy(t_np).view(torch.float16)

    @override
    def create_encoding(
        self,
        data: list[Tensor],
        bit_count: int,
        dtype: EncodingDtype,
    ) -> TensorEncoding:
        return MsetEncoding(
            _encoded_data=data,
            _bit_count=bit_count,
            _decoded_tensors=None,
            _dtype=dtype,
        )


@final
@dataclass
class MsetEncoding(InPlaceEncoding):
    """The encoding produced by `MsetEncoder`. See `MsetEncoder` for details."""

    @override
    def decode_float16(self, t: Tensor) -> Tensor:
        encoded_np = t.view(torch.uint16).numpy(force=True).copy()
        mset.decode_u16(encoded_np)
        return torch.from_numpy(encoded_np).view(torch.float16)

    @override
    def decode_float32(self, t: Tensor) -> Tensor:
        encoded_np = t.numpy(force=True).copy()
        mset.decode_f32(encoded_np)
        return torch.from_numpy(encoded_np)

    @override
    def clone(self) -> MsetEncoding:
        copied_data = [t.clone() for t in self._encoded_data]
        if self._decoded_tensors is not None:
            copied_decoded = [t.clone() for t in self._decoded_tensors]
        else:
            copied_decoded = None

        return MsetEncoding(
            copied_data,
            self._bit_count,
            copied_decoded,
            self._dtype,
        )
