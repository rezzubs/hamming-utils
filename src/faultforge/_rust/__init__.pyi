import numpy as np
import numpy.typing as npt

type ListOfArray[T: np.generic] = list[npt.NDArray[T]]

class Fault:
    @staticmethod
    def stuck_at_0() -> Fault: ...
    @staticmethod
    def stuck_at_1() -> Fault: ...
    @staticmethod
    def flip() -> Fault: ...

class Picker:
    """An iterator which returns numbers from 0..n in a random order until all values are consumed.

    Every returned value is unique.

    This is based on the [Fisher-Yates
    shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle) but
    instead of shuffling the whole sequence we just return the target index for
    the swap.
    """

    def __init__(self, size: int, seed: int | None = None) -> None: ...
    @staticmethod
    def from_returned(
        initial_size: int,
        already_returned: set[int],
        seed: int | None = None,
    ) -> Picker:
        """Reconstruct a picker that will not return any of the `already_returned` values.

        The remaining values returned will be a valid permutation of
        `0..initial_size` excluding `already_returned`. Raises `ValueError` if
        any returned value is outside `0..initial_size`.
        """

    def reset(self) -> None:
        """Reset the picker to its initial state."""

    @property
    def initial_size(self) -> int:
        """The initial size of the picker, as passed to the constructor."""

    @property
    def size(self) -> int:
        """The number of remaining values."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Picker: ...
    def __next__(self) -> int: ...

def list_of_array_fault_f32(
    input: ListOfArray[np.float32],
    fault: Fault,
    target_bit: int,
) -> None: ...
def list_of_array_fault_u16(
    input: ListOfArray[np.uint16],
    fault: Fault,
    target_bit: int,
) -> None: ...
def list_of_array_fault_u8(
    input: ListOfArray[np.uint8],
    fault: Fault,
    target_bit: int,
) -> None: ...
def list_of_array_faults_f32(
    input: ListOfArray[np.float32],
    faults: list[tuple[Fault, int]],
) -> None: ...
def list_of_array_faults_u16(
    input: ListOfArray[np.uint16],
    faults: list[tuple[Fault, int]],
) -> None: ...
def list_of_array_faults_u8(
    input: ListOfArray[np.uint8],
    faults: list[tuple[Fault, int]],
) -> None: ...
