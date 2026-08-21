"""Progress reporting for long-running operations.

See `faultforge.progress` for a general overview.
"""

import abc
import logging
import threading
import time
from types import TracebackType
from typing import Self, final, override

logger = logging.getLogger(__name__)

DEFAULT_MIN_LOG_INTERVAL = 5.0
"""Default minimum number of seconds between progress log lines for a stage."""


class StageHandle(abc.ABC):
    """A named unit of long-running work.

    Used as a context manager: entering logs that the stage started, exiting
    logs completion with elapsed time. Call `advance` from within a loop to
    report incremental progress; logging is throttled to avoid spamming.

    Obtain one via `stage()`, which is the preferred entry point for call
    sites that accept an optional `Progress` since it handles the `None`
    case.
    """

    @abc.abstractmethod
    def advance(self, n: int = 1) -> None:
        """Report that `n` more units of work within this stage have completed."""
        ...

    def __enter__(self) -> Self:
        return self

    @abc.abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


@final
class NullStageHandle(StageHandle):
    """A no-op `StageHandle`, used when progress reporting is disabled."""

    @override
    def advance(self, n: int = 1) -> None:
        pass

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass


_NULL_STAGE = NullStageHandle()


@final
class ProgressStage(StageHandle):
    """The `StageHandle` implementation returned by `Progress.stage`.

    When `total` is known, `advance` logs a throttled `current/total (%)`
    line. When `total` is `None` the underlying operation is an opaque,
    single blocking call (e.g. a network download or an FFI call with no
    callback hook) that never calls `advance` at all; a background daemon
    thread ticks a "still running" heartbeat every `min_log_interval` seconds
    instead, so the user still sees periodic confirmation of life.
    """

    def __init__(self, name: str, total: int | None, min_log_interval: float) -> None:
        self.name: str = name
        self.total: int | None = total
        self.current: int = 0
        self._min_log_interval: float = min_log_interval
        self._start: float = 0.0
        self._last_log: float = 0.0
        self._stop_heartbeat: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None

    @override
    def __enter__(self) -> Self:
        self._start = time.monotonic()
        self._last_log = self._start
        logger.info(f"{self.name}: started")

        if self.total is None:
            self._stop_heartbeat = threading.Event()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_thread.start()

        return self

    def _heartbeat_loop(self) -> None:
        assert self._stop_heartbeat is not None
        while not self._stop_heartbeat.wait(self._min_log_interval):
            elapsed = time.monotonic() - self._start
            logger.info(f"{self.name}: still running, {elapsed:.1f}s elapsed")

    @override
    def advance(self, n: int = 1) -> None:
        self.current += n
        now = time.monotonic()
        if now - self._last_log < self._min_log_interval:
            return
        self._last_log = now
        elapsed = now - self._start

        if self.total is not None:
            percent = self.current / self.total * 100 if self.total else 100.0
            logger.info(
                f"{self.name}: {self.current}/{self.total} "
                f"({percent:.1f}%), {elapsed:.1f}s elapsed"
            )
        else:
            logger.info(
                f"{self.name}: {self.current} processed, {elapsed:.1f}s elapsed"
            )

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stop_heartbeat is not None:
            self._stop_heartbeat.set()
            assert self._heartbeat_thread is not None
            self._heartbeat_thread.join()

        elapsed = time.monotonic() - self._start
        if exc_type is not None:
            logger.info(f"{self.name}: failed after {elapsed:.1f}s")
            return

        suffix = f" ({self.current}/{self.total})" if self.total is not None else ""
        message = f"{self.name}: done{suffix}, {elapsed:.1f}s elapsed"
        if elapsed > self._min_log_interval:
            logger.info(message)
        else:
            logger.debug(message)


@final
class Progress:
    """Reports on named stages of long-running work via `logging`.

    Pass an instance through call sites that support progress reporting; pass
    `None` (the default everywhere it's accepted) to disable reporting with
    no overhead. Use the module-level `stage` function rather than calling
    `.stage` directly at any call site where the `Progress` may be `None` -
    it handles the `None` case for you.
    """

    __slots__ = ("min_log_interval",)

    def __init__(self, min_log_interval: float = DEFAULT_MIN_LOG_INTERVAL) -> None:
        self.min_log_interval: float = min_log_interval

    def stage(self, name: str, total: int | None = None) -> StageHandle:
        """Start a named stage of work.

        `total` is the number of units of work if known ahead of time (enables
        `current/total (%)` reporting); `None` means unknowable, which falls
        back to a periodic "still running" heartbeat logged while the stage
        is open.
        """
        return ProgressStage(name, total, self.min_log_interval)


def stage(
    progress: Progress | None, name: str, total: int | None = None
) -> StageHandle:
    """Start a named stage of work, or a no-op if `progress` is `None`.

    This is the low-boilerplate entry point for call sites that accept an
    optional `Progress`:

        with stage(self._progress, "Loading model") as s:
            ...
            s.advance()
    """
    if progress is None:
        return _NULL_STAGE
    return progress.stage(name, total)
