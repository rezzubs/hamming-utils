import numpy as np
import numpy.typing as npt
from faultforge._rust import Fault

type ListOfArray[T: np.generic] = list[npt.NDArray[T]]

class Encoding:
    def decode_f32(
        self,
    ) -> tuple[ListOfArray[np.float32], list[bool]]: ...
    def decode_u16(
        self,
    ) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
    def apply_fault(self, fault: Fault, target_bit: int) -> None: ...
    def apply_faults(self, faults: list[tuple[Fault, int]]) -> None: ...
    def clone(self) -> Encoding: ...
    def bit_count(self) -> int: ...

def encode_f32(
    input: ListOfArray[np.float32],
    bits_per_chunk: int,
) -> Encoding: ...
def encode_u16(
    input: ListOfArray[np.uint16],
    bits_per_chunk: int,
) -> Encoding: ...
def decode_f32(
    encoded: npt.NDArray[np.uint8],
    encoded_bit_count: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.float32], list[bool]]: ...
def decode_u16(
    encoded: npt.NDArray[np.uint8],
    encoded_bit_count: int,
    bits_per_chunk: int,
    decoded_array_element_counts: list[int],
) -> tuple[ListOfArray[np.uint16], list[bool]]: ...
