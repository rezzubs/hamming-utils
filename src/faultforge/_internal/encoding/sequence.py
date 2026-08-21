"""Sequencing encoders."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from torch import Tensor

from faultforge._internal.encoding.abc import (
    Encoder,
    Encoding,
    TensorEncoder,
    TensorEncoding,
)
from faultforge._internal.fault import Fault
from faultforge._internal.fingerprint import Fingerprint
from faultforge._internal.progress import Progress


@dataclass(slots=True)
class EncoderSequence(Encoder):
    """An encoder which composes other encoders by applying them sequentially.

    The first encoder of `head` will be applied first, followed by the second,
    and so on. The `tail` encoder will be applied last.
    """

    head: Sequence[TensorEncoder]
    tail: Encoder

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(
            kind="sequence",
            children={
                "head": [encoder.fingerprint() for encoder in self.head],
                "tail": [self.tail.fingerprint()],
            },
        )

    @override
    def encode(self, ts: list[Tensor], *, progress: Progress | None = None) -> Encoding:
        head_encodings: list[TensorEncoding] = []

        for encoder in self.head:
            encoding = encoder.encode(ts, progress=progress)
            head_encodings.append(encoding)
            ts = encoding.encoded_tensors()

        tail_encoding = self.tail.encode(ts, progress=progress)

        return EncodingSequence(head_encodings, tail_encoding)


@dataclass
class EncodingSequence(Encoding):
    """An encoding which is created by applying multiple encoders in sequence."""

    _head: list[TensorEncoding]
    _tail: Encoding

    @override
    def decode(self) -> list[Tensor]:
        ts = self._tail.decode()

        for encoding in reversed(self._head):
            for original, updated in zip(encoding.encoded_tensors(), ts, strict=True):
                _ = original.copy_(updated)
            encoding.trigger_recompute()

            ts = encoding.decode()

        return ts

    @override
    def apply_fault(self, fault: Fault, target_bit: int) -> None:
        self._tail.apply_fault(fault, target_bit)

    @override
    def apply_faults(self, faults: Sequence[tuple[Fault, int]]) -> None:
        self._tail.apply_faults(faults)

    @override
    def bit_count(self) -> int:
        return self._tail.bit_count()

    @override
    def clone(self) -> EncodingSequence:
        return EncodingSequence([h.clone() for h in self._head], self._tail.clone())
