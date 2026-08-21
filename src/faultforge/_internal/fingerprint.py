"""Structural fingerprints for identifying experiments and their components."""

from dataclasses import dataclass
from typing import override

from pydantic import BaseModel, ConfigDict

type Scalar = str | int | float | bool | None
"""A semantic value simple enough to live directly inside a `Fingerprint`."""


class Fingerprint(BaseModel):
    """A structural, serializable identity for an experiment component.

    It's a small, serializable description of the *semantics* of an
    experiment component (an encoder, a model/dataset bundle, ...). It
    captures only the parts that change what the component does.
    Environmental knobs like the device, batch size or filesystem paths
    should be deliberately left out.

    Fingerprints are never turned back into live objects. They exist purely
    so that a result file saved to disk can be matched against the
    configuration that is about to be used, and so that a mismatch can be
    reported in a way that points at exactly what changed.

    Each component produces a node describing itself. Nodes nest, so composite
    components (such as an encoder that sequences other encoders) are described
    by nesting the fingerprints of their parts under `children`.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    """A discriminator naming the component, e.g. ``"secded"`` or ``"sequence"``."""
    scalars: dict[str, Scalar] = {}
    """Semantic scalar parameters, e.g. ``{"bits_per_chunk": 8}``."""
    children: dict[str, list[Fingerprint]] = {}
    """Named groups of nested fingerprints.

    Every group is a list so that components holding an arbitrary number of
    sub-components are described uniformly. A single child is a one-element list.
    """

    def diff(self, other: Fingerprint) -> list[FingerprintDifference]:
        """Return the differences that make `other` differ from `self`.

        An empty list means the two fingerprints are structurally identical. The
        differences are path-qualified so they can be shown to a user verbatim.
        `self` is treated as the expected fingerprint and `other` as the actual
        one.
        """
        differences: list[FingerprintDifference] = []
        collect_differences(self, other, [], differences)
        return differences

    def raise_if_differs(self, other: Fingerprint) -> None:
        """Raise `FingerprintError` if `other` differs from `self`.

        Convenience wrapper around `diff`, e.g. for `Experiment.deserialize`
        implementations verifying a loaded fingerprint against the current one.
        """
        if differences := self.diff(other):
            raise FingerprintError(differences)


class Absent:
    """Marks a value that is present on one side of a diff but not the other."""

    @override
    def __repr__(self) -> str:
        return "<absent>"


ABSENT = Absent()
"""Singleton used in a `FingerprintDifference` when a value exists on only one side."""


@dataclass(frozen=True, slots=True)
class FingerprintDifference:
    """A single, localized way in which two fingerprints disagree."""

    path: str
    """Dotted path to the differing location, e.g. ``"encoder.head[0].bits_per_chunk"``."""
    expected: object
    """The value on the expected side, or `ABSENT` if it only exists on the actual side."""
    actual: object
    """The value on the actual side, or `ABSENT` if it only exists on the expected side."""
    description: str
    """A short human-readable explanation of the disagreement."""

    @override
    def __str__(self) -> str:
        return f"{self.path}: {self.description} (expected {self.expected!r}, got {self.actual!r})"


def format_differences(differences: list[FingerprintDifference]) -> str:
    """Render a list of differences as an indented, one-per-line block."""
    return "\n".join(f"  - {difference}" for difference in differences)


class FingerprintError(ValueError):
    """Raised when a saved `Fingerprint` doesn't match the current configuration."""

    differences: list[FingerprintDifference]
    """The differences that caused the mismatch."""

    def __init__(self, differences: list[FingerprintDifference]) -> None:
        self.differences = differences
        super().__init__(
            "Saved experiment does not match the current configuration:\n"
            + format_differences(differences)
        )


def collect_differences(
    expected: Fingerprint,
    actual: Fingerprint,
    path: list[str],
    out: list[FingerprintDifference],
) -> None:
    """Walk two fingerprints in lockstep, appending every disagreement to `out`."""
    location = format_path(path)

    if expected.kind != actual.kind:
        # Differing kinds make the rest of the subtree incomparable, so report
        # the kind and stop descending here.
        out.append(
            FingerprintDifference(
                path=location,
                expected=expected.kind,
                actual=actual.kind,
                description="different component kind",
            )
        )
        return

    for key in sorted(expected.scalars.keys() | actual.scalars.keys()):
        in_expected = key in expected.scalars
        in_actual = key in actual.scalars
        scalar_path = format_path([*path, key])

        if in_expected and in_actual:
            if expected.scalars[key] != actual.scalars[key]:
                out.append(
                    FingerprintDifference(
                        path=scalar_path,
                        expected=expected.scalars[key],
                        actual=actual.scalars[key],
                        description="changed parameter",
                    )
                )
        elif in_expected:
            out.append(
                FingerprintDifference(
                    path=scalar_path,
                    expected=expected.scalars[key],
                    actual=ABSENT,
                    description="missing parameter",
                )
            )
        else:
            out.append(
                FingerprintDifference(
                    path=scalar_path,
                    expected=ABSENT,
                    actual=actual.scalars[key],
                    description="unexpected parameter",
                )
            )

    for key in sorted(expected.children.keys() | actual.children.keys()):
        expected_group = expected.children.get(key)
        actual_group = actual.children.get(key)
        group_path = format_path([*path, key])

        # only one of the two branches will match at a time.
        if expected_group is None:
            out.append(
                FingerprintDifference(
                    path=group_path,
                    expected=ABSENT,
                    actual=f"{len(actual_group or [])} component(s)",
                    description="unexpected component group",
                )
            )
            continue
        if actual_group is None:
            out.append(
                FingerprintDifference(
                    path=group_path,
                    expected=f"{len(expected_group)} component(s)",
                    actual=ABSENT,
                    description="missing component group",
                )
            )
            continue

        if len(expected_group) != len(actual_group):
            out.append(
                FingerprintDifference(
                    path=group_path,
                    expected=len(expected_group),
                    actual=len(actual_group),
                    description="different number of components",
                )
            )

        for index in range(min(len(expected_group), len(actual_group))):
            collect_differences(
                expected_group[index],
                actual_group[index],
                [*path, f"{key}[{index}]"],
                out,
            )


def format_path(path: list[str]) -> str:
    """Join path segments into a dotted location, defaulting to the root marker."""
    return ".".join(path) if path else "<root>"
