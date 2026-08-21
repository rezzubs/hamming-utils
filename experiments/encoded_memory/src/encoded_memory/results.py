"""Turning saved `encoded_memory` result files into comparable configurations.

This is the data-wrangling half of the `compare`/`heatmap` commands (the
figure-drawing half lives in `plots`). It has no `matplotlib` dependency, so
it can be exercised without a plotting backend.

Vocabulary used across this module and `plots`:

- **configuration** - a set of loaded results that are identical in every
  fingerprint field *except* the injected fault count. They only differ in bit
  error rate, so they describe "the same thing measured at several rates" and
  are drawn as a single line. This is the unit `compare` compares. Grouping is
  driven entirely by fingerprint equality, never by how files are laid out on
  disk.
- **bit error rate** - `faults / total_bits`; the x-axis of `compare`. The one
  quantity allowed to vary within a configuration.
- **split key** (`plots.GroupBy`) - a separate concept: a fingerprint-derived
  key (dtype, model, ...) that `compare` uses to fan a single chart out into a
  *grid* of rows and/or columns via `row_by`/`col_by`. Splitting the grid is
  unrelated to which configuration a result belongs to.

The `compare` pipeline, end to end:

1. `commands.compare` parses each path argument into a path plus an optional
   `=LABEL` override.
2. `load_results` discovers and loads every file under those paths into one
   flat pool of `(path, result)` pairs, skipping anything that fails to load.
3. `build_configurations` clusters that pool into `Configuration`s and names
   each one.
4. `plots.build_compare_figure` places each configuration in its grid cell and
   draws its line.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from encoded_memory import SavedResult
from faultforge import Fingerprint

logger = logging.getLogger(__name__)


def discover_result_files(path: Path) -> list[Path]:
    """Resolve `path` to the result file(s) it names.

    A file resolves to itself; a directory resolves to every regular file
    found recursively within it, sorted for determinism.
    """
    resolved = path.expanduser()
    if resolved.is_dir():
        return sorted(p for p in resolved.rglob("*") if p.is_file())
    return [resolved]


def load_results(paths: Sequence[Path]) -> list[tuple[Path, SavedResult]]:
    """Step 2 of the `compare` pipeline: load every result file found across
    all of `paths` (recursing into any directories) into one flat pool of
    `(path, result)` pairs.

    Skips (with a `logger.warning`) any file that isn't a valid saved
    result - e.g. a stray non-result file living alongside real ones in a
    directory - rather than failing the whole load.
    """
    loaded: list[tuple[Path, SavedResult]] = []
    for path in paths:
        for file in discover_result_files(path):
            try:
                loaded.append((file, SavedResult.load(file)))
            except Exception as error:
                logger.warning(
                    f"Failed to load a result from {file} - skipping\n-> {error}"
                )
    return loaded


_VARYING_SCALAR = "faults"
"""The one `Fingerprint` scalar clustering ignores: it's expected to vary
within a configuration, since it's the x-axis being compared (bit error rate)."""


@dataclass(slots=True, frozen=True)
class Configuration:
    """One line on a `compare` chart: results that match on everything except
    bit error rate, plus the legend label to draw them under.
    """

    label: str
    results: list[tuple[Path, SavedResult]]


def _fingerprint_without_faults(fingerprint: Fingerprint) -> Fingerprint:
    return fingerprint.model_copy(
        update={
            "scalars": {
                key: value
                for key, value in fingerprint.scalars.items()
                if key != _VARYING_SCALAR
            }
        }
    )


def _cluster_by_fingerprint(
    loaded: Sequence[tuple[Path, SavedResult]],
) -> list[list[tuple[Path, SavedResult]]]:
    """Group entries whose fingerprints agree on everything except `faults`.

    Driven entirely by `Fingerprint` structural equality (via
    `Fingerprint.diff`), not by how the caller organized files on disk.
    """
    clusters: list[tuple[Fingerprint, list[tuple[Path, SavedResult]]]] = []
    for path, result in loaded:
        key = _fingerprint_without_faults(result.fingerprint)
        for representative, entries in clusters:
            if not representative.diff(key):
                entries.append((path, result))
                break
        else:
            clusters.append((key, [(path, result)]))
    return [entries for _, entries in clusters]


def build_configurations(
    loaded: Sequence[tuple[Path, SavedResult]],
    *,
    label_overrides: Mapping[Path, str],
) -> list[Configuration]:
    """Step 3 of the `compare` pipeline: cluster `loaded` into configurations
    and label each one.

    A configuration's label is whichever `label_overrides` entry applies to
    one of its files - warning (and keeping the first one seen) if more than
    one distinct override applies to the same configuration, since that means
    the caller gave conflicting names to what turned out to be one line. Falls
    back to the stem of the configuration's first (sorted) file when no
    override applies.
    """
    configurations: list[Configuration] = []
    for cluster in _cluster_by_fingerprint(loaded):
        cluster_sorted = sorted(cluster, key=lambda entry: entry[0])

        label: str | None = None
        for path, _ in cluster_sorted:
            override = label_overrides.get(path)
            if override is None:
                continue
            if label is not None and label != override:
                logger.warning(
                    f"conflicting labels {label!r} and {override!r} apply to "
                    "the same configuration; keeping the first"
                )
                continue
            label = override

        if label is None:
            label = cluster_sorted[0][0].stem

        configurations.append(Configuration(label=label, results=cluster_sorted))
    return configurations


def configuration_points(
    configuration: Configuration, *, percentile: float | None
) -> list[tuple[float, float]]:
    """Flatten every run across every file in `configuration` into
    `(bit_error_rate, score)` points, one per distinct bit error rate.

    Runs recorded at the same bit error rate (whether from the same file or
    different files within the configuration) are pooled and reduced to
    `percentile` (or the mean, if `None`). Sorted by bit error rate.
    """
    scores_by_rate: dict[float, list[float]] = {}
    for _, result in configuration.results:
        rate = result.bit_error_rate()
        scores_by_rate.setdefault(rate, []).extend(result.scores())

    def reduce(scores: list[float]) -> float:
        if percentile is None:
            return float(np.mean(scores))
        return float(np.percentile(scores, percentile))

    return sorted((rate, reduce(scores)) for rate, scores in scores_by_rate.items())


def bit_position_histogram(
    bitmask: Sequence[int], *, skip_multi_bit: bool = False
) -> dict[int, int]:
    """Count how many `bitmask` elements had each bit position flipped.

    Maps bit position (0-indexed from the LSB) to how many elements had that
    position differ from the golden value, by decomposing each stored xor value
    bit-by-bit. This is the "which bit index gets hit" view the `heatmap`
    command's bottom panel is built from.

    Pass `skip_multi_bit=True` to exclude elements with more than one faulty
    bit, isolating cases where a single bit flip's position is unambiguous.
    """
    histogram: dict[int, int] = {}
    for value in bitmask:
        if skip_multi_bit and value.bit_count() > 1:
            continue
        position = 0
        remaining = value
        while remaining:
            if remaining & 1:
                histogram[position] = histogram.get(position, 0) + 1
            remaining >>= 1
            position += 1
    return histogram
