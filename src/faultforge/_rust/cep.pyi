from typing import ClassVar

import numpy as np
import numpy.typing as npt

class Scheme:
    D3P1: ClassVar[Scheme]
    D7P1: ClassVar[Scheme]
    D15P1: ClassVar[Scheme]

def encode_f32(arr: npt.NDArray[np.float32], scheme: Scheme) -> None: ...
def decode_f32(arr: npt.NDArray[np.float32], scheme: Scheme) -> None: ...
def encode_u16(arr: npt.NDArray[np.uint16], scheme: Scheme) -> None: ...
def decode_u16(arr: npt.NDArray[np.uint16], scheme: Scheme) -> None: ...
