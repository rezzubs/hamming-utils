"""Tests for the Fingerprint structural diff (faultforge.fingerprint)."""

from faultforge import Fingerprint
from faultforge.fingerprint import (
    ABSENT,
    format_differences,
)


def test_identical_fingerprints_have_no_differences():
    fingerprint = Fingerprint(kind="secded", scalars={"bits_per_chunk": 8})
    assert fingerprint.diff(fingerprint) == []


def test_equality_matches_structural_diff():
    one = Fingerprint(kind="cep", scalars={"scheme": "d3p1"})
    same = Fingerprint(kind="cep", scalars={"scheme": "d3p1"})
    assert one == same
    assert one.diff(same) == []


def test_different_kind_reports_kind_and_stops_descending():
    expected = Fingerprint(kind="mset", scalars={"a": 1})
    actual = Fingerprint(kind="secded", scalars={"a": 2})

    differences = expected.diff(actual)

    assert len(differences) == 1
    difference = differences[0]
    assert difference.path == "<root>"
    assert difference.expected == "mset"
    assert difference.actual == "secded"
    # The differing scalar is not reported because the subtree is incomparable.
    assert "different component kind" in difference.description


def test_changed_scalar():
    expected = Fingerprint(kind="secded", scalars={"bits_per_chunk": 8})
    actual = Fingerprint(kind="secded", scalars={"bits_per_chunk": 16})

    differences = expected.diff(actual)

    assert len(differences) == 1
    assert differences[0].path == "bits_per_chunk"
    assert differences[0].expected == 8
    assert differences[0].actual == 16


def test_missing_scalar_uses_absent_on_the_actual_side():
    expected = Fingerprint(kind="secded", scalars={"bits_per_chunk": 8})
    actual = Fingerprint(kind="secded")

    differences = expected.diff(actual)

    assert len(differences) == 1
    assert differences[0].expected == 8
    assert differences[0].actual is ABSENT


def test_unexpected_scalar_uses_absent_on_the_expected_side():
    expected = Fingerprint(kind="secded")
    actual = Fingerprint(kind="secded", scalars={"bits_per_chunk": 8})

    differences = expected.diff(actual)

    assert len(differences) == 1
    assert differences[0].expected is ABSENT
    assert differences[0].actual == 8


def test_missing_child_group():
    expected = Fingerprint(
        kind="sequence", children={"tail": [Fingerprint(kind="mset")]}
    )
    actual = Fingerprint(kind="sequence")

    differences = expected.diff(actual)

    assert len(differences) == 1
    assert differences[0].path == "tail"
    assert differences[0].actual is ABSENT


def test_unexpected_child_group():
    expected = Fingerprint(kind="sequence")
    actual = Fingerprint(kind="sequence", children={"head": [Fingerprint(kind="mset")]})

    differences = expected.diff(actual)

    assert len(differences) == 1
    assert differences[0].path == "head"
    assert differences[0].expected is ABSENT


def test_different_child_group_length_reports_count_and_common_prefix():
    expected = Fingerprint(
        kind="sequence",
        children={
            "head": [
                Fingerprint(kind="mset"),
                Fingerprint(kind="secded", scalars={"bits_per_chunk": 8}),
            ]
        },
    )
    actual = Fingerprint(
        kind="sequence",
        children={
            "head": [Fingerprint(kind="mset")],
        },
    )

    differences = expected.diff(actual)

    # One difference for the length, and the shared first element still compares
    # equal so it adds nothing.
    assert len(differences) == 1
    assert differences[0].path == "head"
    assert differences[0].expected == 2
    assert differences[0].actual == 1


def test_nested_difference_path_is_indexed():
    expected = Fingerprint(
        kind="sequence",
        children={
            "head": [
                Fingerprint(kind="mset"),
                Fingerprint(kind="secded", scalars={"bits_per_chunk": 8}),
            ],
            "tail": [Fingerprint(kind="cep", scalars={"scheme": "d3p1"})],
        },
    )
    actual = Fingerprint(
        kind="sequence",
        children={
            "head": [
                Fingerprint(kind="mset"),
                Fingerprint(kind="secded", scalars={"bits_per_chunk": 16}),
            ],
            "tail": [Fingerprint(kind="cep", scalars={"scheme": "d7p1"})],
        },
    )

    paths = {difference.path for difference in expected.diff(actual)}

    assert paths == {"head[1].bits_per_chunk", "tail[0].scheme"}


def test_multiple_differences_are_all_collected():
    expected = Fingerprint(
        kind="memory_fi",
        children={
            "bundle": [Fingerprint(kind="cifar", scalars={"model": "resnet20"})],
            "encoder": [Fingerprint(kind="secded", scalars={"bits_per_chunk": 8})],
        },
    )
    actual = Fingerprint(
        kind="memory_fi",
        children={
            "bundle": [Fingerprint(kind="cifar", scalars={"model": "resnet32"})],
            "encoder": [Fingerprint(kind="secded", scalars={"bits_per_chunk": 16})],
        },
    )

    paths = {difference.path for difference in expected.diff(actual)}

    assert paths == {"bundle[0].model", "encoder[0].bits_per_chunk"}


def test_diff_is_directional():
    expected = Fingerprint(kind="secded", scalars={"bits_per_chunk": 8})
    actual = Fingerprint(kind="secded", scalars={"bits_per_chunk": 16})

    forward = expected.diff(actual)[0]
    backward = actual.diff(expected)[0]

    assert (forward.expected, forward.actual) == (8, 16)
    assert (backward.expected, backward.actual) == (16, 8)


def test_serialization_roundtrip_preserves_equality():
    fingerprint = Fingerprint(
        kind="sequence",
        children={
            "head": [Fingerprint(kind="mset")],
            "tail": [Fingerprint(kind="secded", scalars={"bits_per_chunk": 8})],
        },
    )

    restored = Fingerprint.model_validate_json(fingerprint.model_dump_json())

    assert restored == fingerprint
    assert fingerprint.diff(restored) == []


def test_format_differences_renders_one_line_per_difference():
    expected = Fingerprint(kind="cifar", scalars={"model": "resnet20"})
    actual = Fingerprint(kind="cifar", scalars={"model": "resnet32"})

    rendered = format_differences(expected.diff(actual))

    lines = rendered.splitlines()
    assert len(lines) == 1
    assert "model" in lines[0]
    assert "resnet20" in lines[0]
    assert "resnet32" in lines[0]


def test_absent_has_readable_repr():
    assert repr(ABSENT) == "<absent>"
