"""Property tests for MappedLinear: forward output must match plain nn.Linear."""

import copy

import hypothesis.strategies as st
import torch
from faultforge.systolic import MappedLinear, TorchBackend
from hypothesis import given, settings
from torch import nn

_DTYPES = st.sampled_from([torch.float32, torch.float16])

# (W @ x.T).T computes the same math as x @ W.T via a different op sequence,
# not guaranteed bit-identical. float16's default assert_close tolerance is
# too tight for the reordered reduction; loosen it for that dtype only.
# `None` means "use assert_close's dtype-based default".
_TOLERANCES: dict[torch.dtype, tuple[float | None, float | None]] = {
    torch.float32: (None, None),
    torch.float16: (1e-2, 1e-2),
}


@given(
    in_features=st.integers(min_value=1, max_value=16),
    out_features=st.integers(min_value=1, max_value=16),
    batch_size=st.integers(min_value=1, max_value=8),
    bias=st.booleans(),
    dtype=_DTYPES,
)
@settings(max_examples=50)
def test_mapped_linear_matches_plain(
    in_features: int,
    out_features: int,
    batch_size: int,
    bias: bool,
    dtype: torch.dtype,
) -> None:
    module = nn.Linear(in_features, out_features, bias=bias).to(dtype=dtype)
    reference = copy.deepcopy(module)
    mapped = MappedLinear(module, TorchBackend())

    x = torch.randn(batch_size, in_features, dtype=dtype)

    with torch.no_grad():
        expected = reference.forward(x)
        actual = mapped.forward(x)

    rtol, atol = _TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_mapped_linear_exposes_inner() -> None:
    module = nn.Linear(4, 3)
    mapped = MappedLinear(module, TorchBackend())
    assert mapped.inner is module
