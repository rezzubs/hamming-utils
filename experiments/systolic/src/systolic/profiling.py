"""Persisting a `systolic._rust.ProfilingArtifact` to a single `.npz` file.

The metadata sidecar is stored as one more member of the same archive -
the JSON text, viewed as a `uint8` byte array - so the whole artifact
stays one physical file. It's an ordinary array member, indistinguishable
from the other six as far as numpy's `.npz` reader is concerned.
"""

# Chosen over a sidecar file or a wrapping tarball, either of which would
# split the artifact across more than one physical file, and over
# allow_pickle=True (storing metadata as a 0-d object array), a real
# code-execution footgun for untrusted files. Plain numpy.savez_compressed
# needs none of that.

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from faultforge.io import AnyPath
from pydantic import BaseModel

from systolic._rust import ProfilingArtifact

_METADATA_ARRAY_NAME = "metadata_json"
"""Holds `ProfilingMetadata`, JSON-encoded then viewed as a `uint8` byte array.

A plain member of the same `.npz` archive. Any `np.load` caller sees it in
`.files` alongside the six real arrays.
"""

_ARRAY_NAMES = (
    "first_triples",
    "first_fill",
    "active_triples",
    "active_fill",
    "drain_partial_sums",
    "drain_fill",
)


class ProfilingMetadata(BaseModel):
    """Metadata describing one profiling run, saved alongside the arrays."""

    # Deliberately narrow: only fields meaningful at the raw-artifact level.
    # Model/dataset/subsample provenance belongs to a model-level driver
    # built on top of this module and isn't known here.
    array_rows: int
    array_cols: int
    capacity: int
    seed: int


@dataclass(slots=True, frozen=True)
class ProfilingArrays:
    """The six dense arrays making up one profiling artifact.

    Loaded back from disk and named exactly as they're stored.
    """

    first_triples: npt.NDArray[np.float32]
    first_fill: npt.NDArray[np.uintp]
    active_triples: npt.NDArray[np.float32]
    active_fill: npt.NDArray[np.uintp]
    drain_partial_sums: npt.NDArray[np.float32]
    drain_fill: npt.NDArray[np.uintp]


def save_profiling_artifact(
    path: AnyPath, artifact: ProfilingArtifact, metadata: ProfilingMetadata
) -> None:
    """Save one profiling artifact to `path` as a single `.npz` file.

    Writes the six dense arrays plus `metadata` (human-facing only, never
    used for fingerprinting or caching) via one `numpy.savez_compressed`
    call. The file lands at exactly the given `path`.
    """
    resolved = Path(path).expanduser()
    metadata_bytes = np.frombuffer(metadata.model_dump_json().encode(), dtype=np.uint8)
    # Opened as an explicit file handle rather than handing `resolved` straight
    # to `numpy.savez_compressed`: given a bare path, it silently appends
    # `.npz` when missing, which would land the file somewhere other than
    # `resolved`.
    with open(resolved, "wb") as f:
        np.savez_compressed(
            f,
            allow_pickle=False,
            first_triples=artifact.first_triples,
            first_fill=artifact.first_fill,
            active_triples=artifact.active_triples,
            active_fill=artifact.active_fill,
            drain_partial_sums=artifact.drain_partial_sums,
            drain_fill=artifact.drain_fill,
            metadata_json=metadata_bytes,
        )


def load_profiling_artifact(path: AnyPath) -> tuple[ProfilingArrays, ProfilingMetadata]:
    """Load a profiling artifact previously written by `save_profiling_artifact`."""
    resolved = Path(path).expanduser()
    with np.load(resolved, allow_pickle=False) as npz:
        arrays = ProfilingArrays(**{name: npz[name] for name in _ARRAY_NAMES})
        metadata = ProfilingMetadata.model_validate_json(
            bytes(npz[_METADATA_ARRAY_NAME])
        )
    return arrays, metadata
