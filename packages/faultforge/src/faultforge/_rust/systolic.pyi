from typing import ClassVar

import numpy as np
import numpy.typing as npt

class Index2:
    """A coordinate of a processing element in the array."""

    x: int
    y: int

    def __init__(self, x: int, y: int) -> None: ...

class ArrayConfig:
    """The geometry of a systolic array and the width of its data type, as
    needed to compute a register-fault radix."""

    nrows: int
    ncols: int
    dtype_bits: int

    def __init__(self, nrows: int, ncols: int, dtype_bits: int) -> None: ...

class PeRegisterKind:
    """Which register of a processing element a fault targets."""

    Activation: ClassVar[PeRegisterKind]
    Weight: ClassVar[PeRegisterKind]
    Accumulator: ClassVar[PeRegisterKind]

class StuckAtKind:
    """Whether a stuck bit is forced to zero or one."""

    Zero: ClassVar[StuckAtKind]
    One: ClassVar[StuckAtKind]

class Fault:
    """A single fault in array space. Execution-agnostic: it knows how to
    enumerate itself but nothing about how it will be realized (simulated
    vs. lifted)."""

    class Register(Fault):
        """A stuck-at fault in one register of one processing element."""

        target: Index2
        register: PeRegisterKind
        stuck_at: StuckAtKind
        bit_index: int

        def __init__(
            self,
            target: Index2,
            register: PeRegisterKind,
            stuck_at: StuckAtKind,
            bit_index: int,
        ) -> None: ...

    @staticmethod
    def from_id(
        id: int, array: ArrayConfig, allowed_registers: set[PeRegisterKind]
    ) -> Fault:
        """Reconstruct a fault from its id within the (possibly
        register-restricted) fault space over `array`."""

    def to_id(self, array: ArrayConfig, allowed_registers: set[PeRegisterKind]) -> int:
        """This fault's id within the (possibly register-restricted) fault
        space over `array`."""

def fault_radix(array: ArrayConfig, allowed_registers: set[PeRegisterKind]) -> int:
    """The total number of distinct register faults over `array`, restricted
    to `allowed_registers`."""

class AccumulatorFaultPart:
    """The accumulated-fault fix-up for one output row in one pass.
    `for_activations` is `(start, end)`, the half-open range of activation
    rows contributing to the corrupted partial sum."""

    affected_output_row: int
    for_activations: tuple[int, int]

class LiftedFault:
    """A register fault lifted to matrix space: an equivalent description of
    the fault in terms of operations on the weight/activation/output
    matrices rather than the array's physical registers. See
    `docs/fault-lifting.md` for the theory."""

    class Weight(LiftedFault):
        affected_weights: set[Index2]

    class Activation(LiftedFault):
        affected_activation_rows: set[int]
        affected_output_rows: set[int]

    class Accumulator(LiftedFault):
        parts: list[AccumulatorFaultPart]

class Pass:
    """A single pass through the array.

    Describes which activation rows and output rows are connected, and where
    they land in the array. `activation_rows`/`output_rows` are `(start, end)`,
    half-open ranges.
    """

    activation_rows: tuple[int, int]
    output_rows: tuple[int, int]
    array_row_start: int
    array_col_start: int

    def __init__(
        self,
        activation_rows: tuple[int, int],
        output_rows: tuple[int, int],
        array_row_start: int = 0,
        array_col_start: int = 0,
    ) -> None: ...

class Mapping:
    """How a matrix multiplication is mapped onto a systolic array.

    Also the entry point for lifting a register fault to matrix space.
    """

    @staticmethod
    def new(passes: list[Pass]) -> Mapping:
        """Build a mapping from explicit passes.

        Raises `ValueError` if `passes` is empty. Does not itself check that
        `passes` connects every input row to every output row exactly once -
        call `validate()` for that.
        """

    @staticmethod
    def auto_for(
        weights: npt.NDArray[np.float32], array_nrows: int, array_ncols: int
    ) -> Mapping:
        """Automatically map `weights` (`out_features, in_features`) onto an
        `array_nrows x array_ncols` array, splitting into multiple passes if
        the weights don't fit in one."""

    def validate(self) -> None:
        """Confirm that the mapping is valid for running matrix multiplications.

        Checks that every input row is connected to every output row exactly
        once and that row-to-array-index assignments are consistent across
        passes. Raises `ValueError` with the specific problem if not.
        """

    def lift(self, fault: Fault) -> LiftedFault:
        """Lift a register fault targeting this mapping's array to matrix space."""

def simulated_matmul(
    mapping: Mapping,
    weights: npt.NDArray[np.float32],
    activations: npt.NDArray[np.float32],
    array_nrows: int,
    array_ncols: int,
    fault: Fault | None,
) -> npt.NDArray[np.float32]:
    """Run one matmul through a literal, cycle-accurate systolic array
    simulation, optionally with one register fault applied.

    This is the oracle: correct by construction, but simulates the array
    cycle by cycle, so it's slow relative to the lifted torch path. f32 only.
    """
