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
"""Default minimum number of seconds between progress log lines."""


class StageHandle(abc.ABC):
    """A named unit of long-running work.

    Used as a context manager: entering pushes the stage onto its `Progress`'s
    stack of open stages, exiting pops it. Call `advance` from within a loop
    to report incremental progress; logging is throttled to avoid spamming.

    Stages nest: opening one while another is already open (lexically, via
    `with`) makes it a child, and log lines render the whole open chain rather
    than just the innermost stage. Only the outermost (root) stage logs its
    own start/completion; nested stages only ever appear as part of that
    composed line.

    `advance` only ever moves the counter of the stage it's called on. A
    stage's `total` therefore only makes sense when its own caller drives the
    work that produces those units; a stage that merely spans work someone
    else's control flow drives should be opened without a `total`.

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


def _format_duration(seconds: float) -> str:
    """Format `seconds` adaptively: `9.1s`, `2m8s`, or `2h3m`."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes}m{secs}s"
    hours, remainder = divmod(int(seconds), 3600)
    minutes = remainder // 60
    return f"{hours}h{minutes}m"


def _format_frame(frame: ProgressStage, now: float) -> str:
    """Format one stack frame as it appears within a breadcrumb line."""
    duration = _format_duration(now - frame._start)
    if frame.total is not None:
        percent = frame.current / frame.total * 100 if frame.total else 100.0
        return f"{frame.name} {frame.current}/{frame.total} ({percent:.1f}%) {duration}"
    if frame.current:
        return f"{frame.name} {frame.current} {duration}"
    return f"{frame.name} {duration}"


def _format_breadcrumb(stack: list[ProgressStage], now: float) -> str:
    """Format every open stage, outermost first, as one composed line."""
    return " > ".join(_format_frame(frame, now) for frame in stack)


@final
class ProgressStage(StageHandle):
    """The `StageHandle` implementation returned by `Progress.stage`.

    When `total` is known, rendering includes a throttled `current/total (%)`
    fraction. When `total` is `None` the underlying operation is an opaque,
    single blocking call (e.g. a network download or an FFI call with no
    callback hook) that may never call `advance` at all; if this is also the
    root stage, a background daemon thread renders a periodic heartbeat every
    `min_log_interval` seconds instead, so the user still sees periodic
    confirmation of life. Nested stages never get a heartbeat of their own.
    """

    def __init__(self, progress: Progress, name: str, total: int | None) -> None:
        self._progress: Progress = progress
        self.name: str = name
        self.total: int | None = total
        self.current: int = 0
        self._is_root: bool = False
        self._start: float = 0.0
        self._stop_heartbeat: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None

    @override
    def __enter__(self) -> Self:
        self._start = time.monotonic()
        stack = self._progress._stack
        self._is_root = len(stack) == 0
        stack.append(self)

        if self._is_root:
            self._progress._last_render = self._start
            logger.info(f"{self.name}: started")

            if self.total is None:
                self._stop_heartbeat = threading.Event()
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop, daemon=True
                )
                self._heartbeat_thread.start()
        else:
            # A newly opened nested stage changes the breadcrumb (a new
            # frame appears), so it's worth an attempt even though nothing
            # has advanced yet; still subject to the shared throttle.
            self._progress._render()

        return self

    def _heartbeat_loop(self) -> None:
        assert self._stop_heartbeat is not None
        while not self._stop_heartbeat.wait(self._progress.min_log_interval):
            self._progress._render()

    @override
    def advance(self, n: int = 1) -> None:
        self.current += n
        self._progress._render()

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        _ = exc
        _ = tb

        if self._stop_heartbeat is not None:
            self._stop_heartbeat.set()
            assert self._heartbeat_thread is not None
            self._heartbeat_thread.join()

        elapsed = time.monotonic() - self._start
        if exc_type is not None:
            if self._is_root:
                logger.info(f"{self.name}: failed after {_format_duration(elapsed)}")
            self._progress._stack.pop()
            return

        if self._is_root:
            suffix = f" ({self.current}/{self.total})" if self.total is not None else ""
            message = f"{self.name}: done{suffix}, {_format_duration(elapsed)} elapsed"
            if elapsed > self._progress.min_log_interval:
                logger.info(message)
            else:
                logger.debug(message)
        else:
            # Render before popping, so the completed leaf still appears with
            # its final duration; still subject to the shared throttle, so a
            # fast-moving stack of short-lived stages doesn't spam a line per
            # stage.
            self._progress._render()

        self._progress._stack.pop()


@final
class Progress:
    """Reports on named stages of work via `logging`, as one throttled
    breadcrumb line per currently-open chain of nested stages.

    Pass an instance through call sites that support progress reporting; pass
    `None` (the default everywhere it's accepted) to disable reporting with
    no overhead. Use the module-level `stage` function rather than calling
    `.stage` directly at any call site where the `Progress` may be `None` -
    it handles the `None` case for you.
    """

    __slots__ = ("min_log_interval", "_stack", "_last_render")

    def __init__(self, min_log_interval: float = DEFAULT_MIN_LOG_INTERVAL) -> None:
        self.min_log_interval: float = min_log_interval
        self._stack: list[ProgressStage] = []
        self._last_render: float = 0.0

    def stage(self, name: str, total: int | None = None) -> StageHandle:
        """Start a named stage of work, nested under whichever stage (if any)
        is already open on this `Progress`.

        `total` is the number of units of work if known ahead of time (enables
        `current/total (%)` reporting); `None` means unknowable, which falls
        back to a periodic heartbeat if this becomes the root stage.
        """
        return ProgressStage(self, name, total)

    def _render(self) -> None:
        """Log the current stack as one throttled breadcrumb line."""
        if not self._stack:
            return
        now = time.monotonic()
        if now - self._last_render < self.min_log_interval:
            return
        self._last_render = now
        logger.info(_format_breadcrumb(self._stack, now))


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
