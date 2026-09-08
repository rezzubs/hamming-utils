from typing import ClassVar

import numpy as np
import numpy.typing as npt

class Index2:
    """A coordinate of a processing element in the array."""

    x: int
    y: int

    def __init__(self, x: int, y: int) -> None: ...

class ArrayConfig:
    """The geometry of a systolic array and its data type width.

    Used to compute a register-fault radix.
    """

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
    """A single fault in array space.

    Execution-agnostic: it knows how to enumerate itself but nothing
    about how it will be realized (simulated vs. lifted).
    """

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
        """Reconstruct a fault from its id.

        The id is within the (possibly register-restricted) fault space
        over `array`.
        """

    def to_id(self, array: ArrayConfig, allowed_registers: set[PeRegisterKind]) -> int:
        """Return this fault's id within the fault space over `array`.

        The space may be restricted to `allowed_registers`.
        """

def fault_radix(array: ArrayConfig, allowed_registers: set[PeRegisterKind]) -> int:
    """Return the total number of distinct register faults over `array`.

    Restricted to `allowed_registers`.
    """

class AccumulatorFaultPart:
    """The accumulated-fault fix-up for one output row in one pass.

    `for_activations` is `(start, end)`, the half-open range of
    activation rows contributing to the corrupted partial sum.
    """

    affected_output_row: int
    for_activations: tuple[int, int]

class LiftedFault:
    """A register fault lifted to matrix space.

    An equivalent description of the fault in terms of operations on
    the weight/activation/output matrices rather than the array's
    physical registers. See `docs/fault-lifting.md` for the theory.
    """

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
        """Automatically map `weights` onto an array.

        `weights` is `(out_features, in_features)`; the array is
        `array_nrows x array_ncols`. Splits into multiple passes if the
        weights don't fit in one.
        """

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
    """Run one matmul through a cycle-accurate systolic array simulation.

    Optionally applies one register fault. This is the oracle: correct
    by construction, but simulates the array cycle by cycle, so it's
    slow relative to the lifted torch path. f32 only.
    """

class ProfilingArtifact:
    """The dense, on-disk-ready result of one or more profiled matmuls.

    Produced by `Profiler.artifact`. Each `*_fill` entry is
    `min(capacity, observations)` for that PE - the number of real
    samples at the front of the matching `*_triples`/
    `drain_partial_sums` entry; anything at or past `fill` along the
    sample axis is padding (`0.0`), not a real observation. ZERO-regime
    PEs store nothing (all real inputs there are structurally zero).
    """

    first_triples: npt.NDArray[np.float32]
    """Shape `(array_nrows, array_ncols, capacity, 3)`.

    Per-PE FIRST-regime `(activation, weight, partial_sum)` samples.
    """
    first_fill: npt.NDArray[np.uintp]
    """Shape `(array_nrows, array_ncols)`.

    Real sample count per PE in `first_triples`.
    """
    active_triples: npt.NDArray[np.float32]
    """Same shape/meaning as `first_triples`, for the ACTIVE regime."""
    active_fill: npt.NDArray[np.uintp]
    """Same shape/meaning as `first_fill`, for `active_triples`."""
    drain_partial_sums: npt.NDArray[np.float32]
    """Shape `(array_nrows, array_ncols, capacity)`.

    Per-PE DRAIN-regime incoming partial-sum samples. Activation/weight
    are always zero in DRAIN, so only the partial sum is kept.
    """
    drain_fill: npt.NDArray[np.uintp]
    """Shape `(array_nrows, array_ncols)`.

    Real sample count per PE in `drain_partial_sums`.
    """

class Profiler:
    """Accumulates a profiling artifact across many matmul calls.

    One call per (layer, batch) pair over a model's forward pass,
    sharing one physical array and one set of per-PE reservoirs across
    all of them.
    """

    def __init__(
        self, array_nrows: int, array_ncols: int, capacity: int, seed: int
    ) -> None:
        """Create a profiler for an `array_nrows x array_ncols` array.

        Every PE's per-regime reservoir is bounded to `capacity`
        samples. A single RNG stream, seeded from `seed`, is shared
        across every PE and regime.
        """

    def run(
        self,
        mapping: Mapping,
        weights: npt.NDArray[np.float32],
        activations: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Run one matmul, accumulating into the shared reservoirs.

        Uses the same cycle-accurate simulation as `simulated_matmul`.
        Returns the plain result so a caller can feed real activations
        through the rest of the model.
        """

    def artifact(self) -> ProfilingArtifact:
        """Snapshot everything accumulated so far into a dense artifact.

        Non-destructive - callable more than once, e.g. to checkpoint
        mid-run.
        """
