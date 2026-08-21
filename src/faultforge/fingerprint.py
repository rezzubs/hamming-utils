"""Diffing helpers for `faultforge.Fingerprint`.

See that class for what a fingerprint is and when to use one; this module
holds the pieces only needed when comparing two fingerprints and reporting
the result.
"""

from faultforge._internal.fingerprint import (
    ABSENT,
    FingerprintDifference,
    FingerprintError,
    Scalar,
    format_differences,
)

__all__ = [
    "ABSENT",
    "FingerprintDifference",
    "FingerprintError",
    "Scalar",
    "format_differences",
]
