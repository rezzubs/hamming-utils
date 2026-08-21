"""Classes for running experiments.

See `faultforge.experiment` for a general overview.
"""

import abc
import logging
import os
import signal
import tempfile
import time
import types
from collections.abc import (
    Callable,
    Sequence,
)
from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path
from typing import (
    IO,
    Self,
)

import scipy.stats

from faultforge._internal.io import AnyPath, is_compressed, open_text

logger = logging.getLogger(__name__)

_COMPRESSED_SUFFIXES = (".zst", ".zstd")


def _warn_on_extension_mismatch(path: AnyPath, *, compressed: bool) -> None:
    """Advisory-only: log a warning if `compressed` disagrees with `path`'s name.

    Never changes what path is actually written to - purely a hint that the
    file's contents and its name disagree about whether it's zstd-compressed,
    which would otherwise confuse tools like `cat`/`jq` reading it later.
    """
    looks_compressed = Path(path).name.endswith(_COMPRESSED_SUFFIXES)
    if compressed and not looks_compressed:
        logger.warning(
            f"Saving compressed data to {path!r}, whose name doesn't end in .zst/.zstd."
        )
    elif not compressed and looks_compressed:
        logger.warning(
            f"Saving uncompressed data to {path!r}, whose name ends in .zst/.zstd."
        )


@dataclass(slots=True)
class SaveConfig:
    """Where and how often `Experiment.run_loop` persists progress."""

    path: AnyPath
    """Where to save the experiment's results, via `Experiment.save_atomic`."""
    interval_seconds: float | None
    """How many seconds between saves. None means save only at the end."""
    compressed: bool = False
    """Whether to save through zstd compression; passed straight through to
    `Experiment.save_atomic`."""


def relative_margin_of_error(
    mean: float | None, margin_of_error: float | None
) -> float | None:
    """The 95% margin of error as a percentage of the mean.

    `None` if either input is `None`. A mean of exactly `0` would otherwise
    raise `ZeroDivisionError` (a legitimate outcome for e.g. a 0% SDC score);
    that case is treated as 0% relative error when there is no error either,
    and as an undefined (infinite) relative error otherwise.
    """
    if mean is None or margin_of_error is None:
        return None
    if mean == 0:
        return 0.0 if margin_of_error == 0 else float("inf")
    return margin_of_error / mean * 100


class ExperimentDisplay:
    """Formats an `Experiment`'s status line for `run_loop`.

    Returned by `Experiment.display`; nothing here is stored on the experiment,
    it's computed on demand. Override any piece to customize; the default
    renders `[Run n]: name = score unit | mean ± moe (95% CI) | Relative MoE: x%
    of mean`, with the `Relative MoE` fragment only shown when a `Stability`
    condition is among the ones currently configured on `run_loop` - it's the
    exact quantity `Stability` checks against its threshold, so it has nothing
    to say if there's no threshold to preview.
    """

    def score_name(self) -> str | None:
        """The name given to the result score, or None to omit it."""
        return None

    def score_unit(self) -> str | None:
        """The unit printed after a score, or None to omit it."""
        return None

    def format_score(self, score: float) -> str:
        """Format a single score value (the latest score, mean, or margin of error)."""
        return f"{score:6.2f}"

    def progress_label(self, run_count: int) -> str:
        """The leading `[...]` progress marker.

        The base case only knows the run count. An experiment that also
        knows a total (e.g. an exhaustive search over a known number of
        cases) should override this using a value it tracks itself, rather
        than have `Experiment` prescribe a "total" concept every subclass
        must carry.
        """
        return f"[Run {run_count}]"

    def extra(self) -> str | None:
        """A string to append as-is to the end of the status message, or None
        to omit it. Include any leading separator/spacing yourself."""
        return None

    def format(
        self,
        *,
        run_count: int,
        score: float,
        mean: float | None,
        margin_of_error: float | None,
        stop_conditions: Sequence[StopCondition] = (),
    ) -> str:
        """Compose the full status line from the pieces above.

        `stop_conditions` is whatever's currently configured on `run_loop`
        (both intrinsic and caller-supplied), passed through so a subclass can
        shape its output around what's actually being checked - the default
        implementation uses it only to decide whether to show `Relative MoE`.
        """
        parts: list[str] = [self.progress_label(run_count), ": "]

        def build() -> None:
            score_name = self.score_name()
            if score_name is not None:
                parts.append(score_name)
                parts.append(" = ")

            score_unit = self.score_unit()

            parts.append(self.format_score(score))
            if score_unit is not None:
                parts.append(score_unit)

            if mean is None:
                return
            parts.append(" | ")
            parts.append(f"mean {self.format_score(mean)}")
            if score_unit is not None:
                parts.append(score_unit)

            if margin_of_error is None:
                return
            parts.append(f" ±{self.format_score(margin_of_error)} (95% CI)")

            has_stability = any(
                isinstance(condition, Stability) for condition in stop_conditions
            )
            if not has_stability:
                return

            relative = relative_margin_of_error(mean, margin_of_error)
            if relative is None:
                return
            parts.append(f" | Relative MoE: {relative:.2f}% of mean")

        build()
        if extra := self.extra():
            parts.append(extra)

        return "".join(parts)


# A check run by `Experiment.run_loop` each iteration, before `run`. Returns a
# human-readable reason to stop, or `None` to keep going.
type StopCondition = Callable[[Experiment], str | None]


class Experiment(abc.ABC):
    """The behavior of a repeatable, scoreable unit of work.

    Subclasses own the shape of their own results (a list, a dict keyed
    however makes sense, ...) and how that state is persisted; this class only
    prescribes what's needed to drive and monitor a series of runs:

    - `run`: perform one iteration and record it internally, any way you like.
    - `scores`: report every recorded score so far, in run order, as plain
      floats. This is the only view the generic machinery below needs, so it
      never has to know your result type.
    - `display`: describe how to format your score, via `ExperimentDisplay`.
    - `serialize` / `deserialize`: turn your own state into a string and back.
      That's the only shape-specific part of persistence; `save`/`save_file`/
      `save_atomic`/`load_from`/`load_from_file` handle the file mechanics
      (atomic writes, path expansion, reading a path or an already-open file,
      optional zstd compression) generically on top of these two.
      Fingerprinting isn't part of this contract either: if you want to
      verify a loaded file against your current configuration, include your
      own `Fingerprint` in `serialize`'s output and check it in `deserialize`
      via `Fingerprint.raise_if_differs`, which raises `FingerprintError` on
      a mismatch.

    `run_loop` drives the experiment: it calls `run` repeatedly, prints progress
    via `format_status`, and stops once any `StopCondition` fires - including
    Ctrl+C, which is just another condition `run_loop` installs internally.
    Conditions come from two places: `stop_conditions()`, overridden by a
    subclass to contribute conditions driven by its own internal state (e.g. "no
    distinct runs remain"), and the `stop_conditions` argument to `run_loop`
    itself, for session-level concerns decided by the caller (e.g. `Stability`,
    `AdditionalRuns`, `MaxRuns`). Neither is required - by default `run_loop`
    runs until interrupted.

    See the `encoded_memory` experiment (`experiments/encoded_memory` in the
    repository) for a complete example.
    """

    @abc.abstractmethod
    def run(self) -> None:
        """Run a single iteration and record its result internally."""

    @abc.abstractmethod
    def scores(self) -> Sequence[float]:
        """Every score recorded so far, in the order the runs happened."""

    def display(self) -> ExperimentDisplay:
        """Describes how `format_status` should render this experiment's score."""
        return ExperimentDisplay()

    def stop_conditions(self) -> Sequence[StopCondition]:
        """Stop conditions intrinsic to this experiment.

        Checked by `run_loop` alongside any passed in by the caller. Override
        to contribute e.g. an exhaustion check driven by your own internal
        state; the default contributes none.
        """
        return ()

    @abc.abstractmethod
    def serialize(self) -> str:
        """Serialize current results to a string, for `save`/`save_atomic`.

        If you want to guard against loading a file saved under a different
        configuration, include your own `Fingerprint` in the output here so
        `deserialize` can check it, e.g. via `Fingerprint.raise_if_differs`.
        """

    @abc.abstractmethod
    def deserialize(self, content: str) -> None:
        """Restore results from a string previously produced by `serialize`."""

    def save(self, path: AnyPath, *, compressed: bool = False) -> None:
        """Save current results to `path`.

        Pass `compressed=True` to write through zstd (`compression.zstd`,
        stdlib) instead of plain text - `serialize()`'s output itself never
        changes, only how it's stored on disk.
        """
        _warn_on_extension_mismatch(path, compressed=compressed)
        with open_text(path, "wt", compressed=compressed) as f:
            self.save_file(f)

    def save_file(self, file: IO[str]) -> None:
        """Save current results to `file`.

        See `save` for a version that takes a path.
        """
        file.write(self.serialize())

    def save_atomic(self, path: AnyPath, *, compressed: bool = False) -> None:
        """Save current results to `path`, atomically.

        Will not corrupt existing data if the write fails partway. Pass
        `compressed=True` to write through zstd instead of plain text, same
        as `save`.
        """
        _warn_on_extension_mismatch(path, compressed=compressed)
        destination = Path(path).expanduser()
        fd, temp_name = tempfile.mkstemp(dir=destination.parent)
        os.close(fd)
        with open_text(temp_name, "wt", compressed=compressed) as temp:
            temp.write(self.serialize())
        os.replace(temp_name, destination)

    def load_from(self, path: AnyPath) -> None:
        """Restore results from `path`, previously written by `save`/`save_atomic`.

        Auto-detects whether `path` is zstd-compressed, so this works on a
        file saved with either `compressed=True` or `compressed=False`
        without the caller needing to know which.

        See `load_from_file` for a version that takes a file-like object.
        """
        with open_text(path, "rt", compressed=is_compressed(path)) as f:
            self.load_from_file(f)

    def load_from_file(self, file: IO[str]) -> None:
        """Restore results from `file`, previously written by `save_file`.

        See `load_from` for a version that takes a path.
        """
        self.deserialize(file.read())

    def run_count(self) -> int:
        """Return the number of recorded runs."""
        return len(self.scores())

    def run_loop(
        self,
        *,
        stop_conditions: Sequence[StopCondition] = (),
        save_config: SaveConfig | None = None,
    ) -> None:
        """Keep running until a stop condition is met, including Ctrl+C."""

        interrupted = _Interrupted()
        all_conditions = [*self.stop_conditions(), *stop_conditions, interrupted]

        dirty = False
        passed_seconds = 0.0
        start = time.monotonic()

        with interrupted:
            while True:
                reason = _first_stop_reason(all_conditions, self)
                if reason is not None:
                    logger.info(reason)
                    break

                self.run()
                print(self.format_status(all_conditions))
                dirty = True

                if save_config is not None and save_config.interval_seconds is not None:
                    now = time.monotonic()
                    passed_seconds += now - start
                    start = now

                    if save_config.interval_seconds < passed_seconds:
                        logger.debug(
                            f"Passed {save_config.interval_seconds}s ({passed_seconds}) since last save"
                        )
                        self.save_atomic(
                            save_config.path, compressed=save_config.compressed
                        )
                        passed_seconds = 0.0

        if dirty and save_config is not None:
            self.save_atomic(save_config.path, compressed=save_config.compressed)

    def margin_of_error(self) -> float | None:
        """Return the margin of error (half-width of the 95% confidence interval)
        for the mean of the current set of scores.

        None if there are less than 2 results.
        """
        scores = self.scores()
        n = len(scores)
        if n < 2:
            return None
        t = scipy.stats.t.ppf(0.975, df=n - 1)
        return float(t * scipy.stats.sem(scores))

    def mean_score(self) -> float | None:
        """Return the mean of the current set of scores.

        None if there are no results yet.
        """
        scores = self.scores()
        if not scores:
            return None
        return float(sum(scores) / len(scores))

    def format_status(
        self, stop_conditions: Sequence[StopCondition] = ()
    ) -> str | None:
        """Formats the current status of the experiment as a str.

        None if there are no results yet. `stop_conditions` lets a caller
        (typically `run_loop`) tell the display what's currently configured;
        it's `()` if called standalone, which `ExperimentDisplay` treats as
        "nothing configured" rather than anything meaningful to report on.
        """
        scores = self.scores()
        if not scores:
            return None
        return self.display().format(
            run_count=len(scores),
            score=scores[-1],
            mean=self.mean_score(),
            margin_of_error=self.margin_of_error(),
            stop_conditions=stop_conditions,
        )


class _Interrupted:
    """A `StopCondition` that fires once Ctrl+C has been received.

    Used as a context manager for the duration of `run_loop`: entering
    installs a SIGINT handler, exiting restores the original - as does the
    first Ctrl+C itself, so a second one behaves normally (e.g. force-quitting).
    """

    def __init__(self) -> None:
        self._triggered = False

    def __enter__(self) -> Self:
        self._original_handler = signal.getsignal(signal.SIGINT)
        _ = signal.signal(signal.SIGINT, self._handle)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        _ = exc_type, exc, tb
        _ = signal.signal(signal.SIGINT, self._original_handler)

    def _handle(self, sig: int, frame: types.FrameType | None) -> None:
        _ = sig, frame
        self._triggered = True
        logger.info(
            "Received Ctrl+C. Finishing current run before stopping. Use Ctrl+C again to force quit."
        )
        _ = signal.signal(signal.SIGINT, self._original_handler)

    def __call__(self, experiment: Experiment) -> str | None:
        _ = experiment
        return "Interrupted by Ctrl+C" if self._triggered else None


def _first_stop_reason(
    conditions: Sequence[StopCondition], experiment: Experiment
) -> str | None:
    """The reason given by the first condition in `conditions` that wants to stop, if any."""
    for condition in conditions:
        if (reason := condition(experiment)) is not None:
            return reason
    return None


@dataclass(slots=True)
class Stability:
    """A `StopCondition`: stop once the mean score's margin of error is small
    relative to the mean."""

    min_samples: int
    """Minimum number of runs before checking the stopping criterion."""
    threshold: float
    """Stop when the relative margin of error (95% margin of error as a percentage of the mean) falls below this value, e.g. 1.0 = 1%."""

    def __call__(self, experiment: Experiment) -> str | None:
        if experiment.run_count() < self.min_samples:
            return None
        relative = relative_margin_of_error(
            experiment.mean_score(), experiment.margin_of_error()
        )
        if relative is not None and relative <= self.threshold:
            return (
                f"Reached stability threshold {self.threshold:.2f}% ({relative:.2f}%)"
            )
        return None


@dataclass(slots=True)
class AdditionalRuns:
    """A `StopCondition`: stop after `count` more runs, on top of however many
    already existed when this instance first got checked (typically the start
    of `run_loop`).

    Use `MaxRuns` instead if you want to cap the *total* run count regardless
    of how many results already exist (e.g. loaded from a save file).
    """

    count: int
    _baseline: int | None = field(default=None, init=False, repr=False)

    def __call__(self, experiment: Experiment) -> str | None:
        if self._baseline is None:
            self._baseline = experiment.run_count()
        if experiment.run_count() - self._baseline >= self.count:
            return f"Reached requested additional run count (+{self.count})"
        return None


@dataclass(slots=True)
class MaxRuns:
    """A `StopCondition`: stop once the total run count reaches `total`,
    including any results that already existed before this instance was ever
    checked (e.g. loaded from a save file).

    Use `AdditionalRuns` instead if you want to run a fixed number more
    regardless of how many results already exist.
    """

    total: int

    def __call__(self, experiment: Experiment) -> str | None:
        if experiment.run_count() >= self.total:
            return f"Reached max run count ({self.total})"
        return None
