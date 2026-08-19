"""The register fault-space configuration."""
from typing import final

from dataclasses import dataclass, field

from faultforge._rust.systolic import ArrayConfig, Fault, PeRegisterKind, fault_radix

_ALL_REGISTERS = frozenset(
    {PeRegisterKind.Activation, PeRegisterKind.Weight, PeRegisterKind.Accumulator}
)

_REGISTER_NAMES = {
    PeRegisterKind.Activation: "activation",
    PeRegisterKind.Weight: "weight",
    PeRegisterKind.Accumulator: "accumulator",
}


@final
@dataclass(frozen=True, slots=True)
class RegisterFaults:
    """A register fault-space config: which registers a fault may target.

    The default is to use all registers (activation, weight, accumulator).
    A different set can be given during construction if you wish to target a
    subset of registers.
    """

    registers: frozenset[PeRegisterKind] = field(default_factory=lambda: _ALL_REGISTERS)

    def radix(self, array: ArrayConfig) -> int:
        """The number of distinct register faults over `array` under this restriction."""
        return fault_radix(array, set(self.registers))

    def fault_from_id(self, id: int, array: ArrayConfig) -> Fault:
        """Reconstruct a fault from its id within `self.radix(array)`."""
        return Fault.from_id(id, array, set(self.registers))

    def fingerprint_scalar(self) -> str:
        """The register subset as a sorted, comma-joined string.

        `Fingerprint.scalars` only accepts flat `str | int | float | bool`
        values, so the subset can't be stored as a native collection; this
        is the encoding used for it there.
        """
        return ",".join(
            sorted(_REGISTER_NAMES[register] for register in self.registers)
        )
