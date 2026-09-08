"""Property tests for MappedConv2d: forward output must match plain nn.Conv2d."""

import copy

import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given, settings
from torch import nn

from systolic import MappedConv2d, TorchBackend

_DTYPES = st.sampled_from([torch.float32, torch.float16])

# im2col+matmul computes the same math as a direct convolution via a
# different op sequence, not guaranteed bit-identical. float16's default
# assert_close tolerance is too tight for the reordered reduction; loosen it
# for that dtype only. `None` means "use assert_close's dtype-based default".
_TOLERANCES: dict[torch.dtype, tuple[float | None, float | None]] = {
    torch.float32: (None, None),
    torch.float16: (1e-2, 1e-2),
}


@given(
    in_channels=st.integers(min_value=1, max_value=4),
    out_channels=st.integers(min_value=1, max_value=4),
    kernel_size=st.integers(min_value=1, max_value=3),
    stride=st.integers(min_value=1, max_value=2),
    padding=st.integers(min_value=0, max_value=2),
    dilation=st.integers(min_value=1, max_value=2),
    batch_size=st.integers(min_value=1, max_value=4),
    input_size=st.integers(min_value=6, max_value=10),
    bias=st.booleans(),
    dtype=_DTYPES,
)
@settings(max_examples=50)
def test_mapped_conv2d_matches_plain(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    batch_size: int,
    input_size: int,
    bias: bool,
    dtype: torch.dtype,
) -> None:
    module = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    ).to(dtype=dtype)
    reference = copy.deepcopy(module)
    mapped = MappedConv2d(module, TorchBackend())

    x = torch.randn(batch_size, in_channels, input_size, input_size, dtype=dtype)

    with torch.no_grad():
        expected = reference.forward(x)
        actual = mapped.forward(x)

    rtol, atol = _TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_mapped_conv2d_rejects_grouped_convolution() -> None:
    module = nn.Conv2d(4, 4, kernel_size=3, groups=2)
    with pytest.raises(ValueError, match="groups"):
        MappedConv2d(module, TorchBackend())


def test_mapped_conv2d_exposes_inner() -> None:
    module = nn.Conv2d(3, 4, kernel_size=3)
    mapped = MappedConv2d(module, TorchBackend())
    assert mapped.inner is module
