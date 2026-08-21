"""Bit-level faults that can be injected into tensors or encoded memory."""

import enum

from faultforge import _rust


class BitFlip:
    """A fault that flips a single bit."""


class StuckAt(enum.Enum):
    """A fault that sets a single bit to a fixed value."""

    Zero = 0
    One = 1


type Fault = BitFlip | StuckAt
"""A fault affecting a single, targeted bit.

Passed together with a bit index to methods like `Encoding.apply_fault` or
`EncodedModule.apply_fault`.
"""


def fault_to_rust(fault: Fault) -> _rust.Fault:
    """Convert a fault to a format accepted by the bindings."""
    match fault:
        case BitFlip():
            return _rust.Fault.flip()
        case StuckAt.Zero:
            return _rust.Fault.stuck_at_0()
        case StuckAt.One:
            return _rust.Fault.stuck_at_1()
