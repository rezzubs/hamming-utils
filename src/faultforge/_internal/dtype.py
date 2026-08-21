"""Data types for various operations."""

from __future__ import annotations

import enum

import torch


class EncodingDtype(enum.StrEnum):
    """Data types that support encoding."""

    F32 = "f32"
    F16 = "f16"

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> EncodingDtype:
        """Convert a torch dtype to an EncodableDtype.

        Raises:
            ValueError: If the dtype is not supported.
        """
        match dtype:
            case torch.float32:
                return cls.F32
            case torch.float16:
                return cls.F16
            case _:
                raise ValueError(f"dtype {dtype} does not support encoding")

    def to_torch(self) -> torch.dtype:
        """Convert an EncodableDtype to a torch dtype."""
        match self:
            case EncodingDtype.F32:
                return torch.float32
            case EncodingDtype.F16:
                return torch.float16

    def bit_count(self) -> int:
        """Return the number of bits in the dtype."""
        match self:
            case EncodingDtype.F32:
                return 32
            case EncodingDtype.F16:
                return 16


class FiDtype(enum.Enum):
    """Data types that support fault injection."""

    F32 = enum.auto()
    F16 = enum.auto()
    U8 = enum.auto()

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> FiDtype:
        """Convert a torch dtype to a FiDtype.

        Raises:
            ValueError: If the dtype is not supported.
        """
        match dtype:
            case torch.float32:
                return cls.F32
            case torch.float16:
                return cls.F16
            case torch.uint8:
                return cls.U8
            case _:
                raise ValueError(f"dtype {dtype} does not support fault injection")

    def to_torch(self) -> torch.dtype:
        """Convert to an equivalent pytorch data type."""
        match self:
            case FiDtype.F32:
                return torch.float32
            case FiDtype.F16:
                return torch.float16
            case FiDtype.U8:
                return torch.uint8

    def bit_width(self) -> int:
        """Return the number of bits in the dtype."""
        match self:
            case FiDtype.F32:
                return 32
            case FiDtype.F16:
                return 16
            case FiDtype.U8:
                return 8
