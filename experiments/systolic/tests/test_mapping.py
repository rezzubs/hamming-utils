"""Tests for hand-built `Mapping`/`Pass` construction, as an alternative to
`Mapping.auto_for`. Exercises the raw `systolic._rust` bindings directly,
since nothing above this layer builds a `Mapping` any other way yet.
"""

import numpy as np
import pytest

from systolic._rust import (
    Fault,
    Index2,
    Mapping,
    Pass,
    PeRegisterKind,
    StuckAtKind,
    simulated_matmul,
)


def test_pass_defaults_offsets_to_zero() -> None:
    p = Pass(activation_rows=(0, 2), output_rows=(0, 3))
    assert p.array_row_start == 0
    assert p.array_col_start == 0
    assert p.activation_rows == (0, 2)
    assert p.output_rows == (0, 3)


def test_mapping_new_rejects_empty() -> None:
    with pytest.raises(ValueError):
        Mapping.new([])


def test_mapping_new_single_pass_matches_auto_for() -> None:
    """A hand-built single-pass mapping covering the whole array should be
    interchangeable with what `auto_for` derives for weights that fit in one
    pass: same matmul output, same fault lift."""
    rng = np.random.default_rng(0)
    out_features, in_features, batch = 3, 4, 2
    array_nrows, array_ncols = in_features, out_features

    weights = rng.standard_normal((out_features, in_features)).astype(np.float32)
    activations = rng.standard_normal((in_features, batch)).astype(np.float32)

    auto = Mapping.auto_for(weights, array_nrows, array_ncols)
    custom = Mapping.new(
        [Pass(activation_rows=(0, in_features), output_rows=(0, out_features))]
    )
    custom.validate()

    expected = simulated_matmul(
        auto, weights, activations, array_nrows, array_ncols, None
    )
    actual = simulated_matmul(
        custom, weights, activations, array_nrows, array_ncols, None
    )
    np.testing.assert_array_equal(actual, expected)

    fault = Fault.Register(
        target=Index2(x=0, y=0),
        register=PeRegisterKind.Weight,
        stuck_at=StuckAtKind.One,
        bit_index=5,
    )
    assert type(custom.lift(fault)) is type(auto.lift(fault))


def test_mapping_validate_detects_missing_connections() -> None:
    """Two passes whose combined activation/output ranges imply a bigger
    universe than either pass alone connects, leaving the cross-connections
    between them unaccounted for."""
    mapping = Mapping.new(
        [
            Pass(activation_rows=(0, 2), output_rows=(0, 2)),
            Pass(activation_rows=(2, 4), output_rows=(2, 4)),
        ]
    )
    with pytest.raises(ValueError):
        mapping.validate()


def test_mapping_validate_detects_duplicate_connections() -> None:
    mapping = Mapping.new(
        [
            Pass(activation_rows=(0, 2), output_rows=(0, 2)),
            Pass(activation_rows=(0, 2), output_rows=(0, 2)),
        ]
    )
    with pytest.raises(ValueError):
        mapping.validate()
