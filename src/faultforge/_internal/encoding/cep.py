"""Chunked Embedded Parity (CEP)."""

import enum
import logging
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor

from faultforge._internal.dtype import EncodingDtype
from faultforge._internal.encoding.abc import (
    InPlaceEncoder,
    InPlaceEncoding,
    TensorEncoding,
)
from faultforge._internal.fingerprint import Fingerprint
from faultforge._rust import cep

_logger = logging.getLogger(__name__)


class CepScheme(enum.StrEnum):
    """How many data bits to use per parity bit.

    D3P1 should result in the best accuracy in most cases because it results in
    the biggest number of chunks per parameter - it can mitigate more faults.
    """

    D3P1 = "d3p1"
    D7P1 = "d7p1"
    D15P1 = "d15p1"

    def _to_rust(self) -> cep.Scheme:
        match self:
            case CepScheme.D3P1:
                return cep.Scheme.D3P1
            case CepScheme.D7P1:
                return cep.Scheme.D7P1
            case CepScheme.D15P1:
                return cep.Scheme.D15P1

    @staticmethod
    def default() -> CepScheme:
        """Return the default scheme."""
        return CepScheme.D3P1


@dataclass
class CepEncoder(InPlaceEncoder):
    """Encodes tensors with Chunked Embedded Parity (CEP).

    The higher bits of each element are chunked based on `scheme` and each
    chunk is given a matching parity bit embedded inside the lower bits. If a
    parity bit doesn't match during decoding, the corresponding chunk is set
    to zero.
    """

    scheme: CepScheme = CepScheme.D3P1

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="cep", scalars={"scheme": self.scheme.value})

    @override
    def encode_float32(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.numpy(force=True)
        cep.encode_f32(t_np, self.scheme._to_rust())
        return torch.from_numpy(t_np)

    @override
    def encode_float16(self, t: Tensor) -> Tensor:
        with torch.no_grad():
            t_np = t.view(torch.uint16).numpy(force=True)
        cep.encode_u16(t_np, self.scheme._to_rust())
        return torch.from_numpy(t_np).view(torch.float16)

    @override
    def create_encoding(
        self,
        data: list[Tensor],
        bit_count: int,
        dtype: EncodingDtype,
    ) -> TensorEncoding:
        return CepEncoding(
            _encoded_data=data,
            _bit_count=bit_count,
            _decoded_tensors=None,
            _dtype=dtype,
            _scheme=self.scheme,
        )


@dataclass
class CepEncoding(InPlaceEncoding):
    """The encoding produced by `CepEncoder`. See `CepEncoder` for details."""

    _scheme: CepScheme

    @override
    def clone(self) -> CepEncoding:
        cloned_data = [t.clone() for t in self._encoded_data]
        if self._decoded_tensors is not None:
            cloned_decoded = [t.clone() for t in self._decoded_tensors]
        else:
            cloned_decoded = None

        return CepEncoding(
            _encoded_data=cloned_data,
            _bit_count=self._bit_count,
            _decoded_tensors=cloned_decoded,
            _dtype=self._dtype,
            _scheme=self._scheme,
        )

    @override
    def decode_float16(self, t: Tensor) -> Tensor:
        encoded_np = t.view(torch.uint16).numpy(force=True).copy()
        cep.decode_u16(encoded_np, self._scheme._to_rust())
        return torch.from_numpy(encoded_np).view(torch.float16)

    @override
    def decode_float32(self, t: Tensor) -> Tensor:
        encoded_np = t.numpy(force=True).copy()
        cep.decode_f32(encoded_np, self._scheme._to_rust())
        return torch.from_numpy(encoded_np)
