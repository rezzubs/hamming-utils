"""Tests for progress reporting (faultforge.progress)."""

import logging
import re
import time

import pytest
from faultforge._internal.progress import ProgressStage, _format_duration
from faultforge.progress import Progress, stage

# The following classes are `_` prefixed to not interpret them as Test classes.


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_stage_logs_started_and_done(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # "done" is only logged at INFO once elapsed time exceeds min_log_interval
    # (see ProgressStage.__exit__); a fake clock simulates a slow stage so
    # this test doesn't need to actually sleep.
    caplog.set_level(logging.INFO)
    clock = _FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    with stage(Progress(min_log_interval=1.0), "X"):
        clock.advance(2.0)

    messages = [r.message for r in caplog.records]
    assert any("X: started" in m for m in messages)
    assert any("X: done" in m for m in messages)


def test_stage_logs_done_at_debug_when_fast(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    clock = _FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    with stage(Progress(min_log_interval=1.0), "X"):
        clock.advance(0.1)

    done_records = [r for r in caplog.records if "X: done" in r.message]
    assert len(done_records) == 1
    assert done_records[0].levelno == logging.DEBUG


def test_stage_reports_total_and_percent(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    total = 4
    with stage(Progress(min_log_interval=0.0), "X", total=total) as s:
        for _ in range(total):
            s.advance()

    messages = [r.message for r in caplog.records]
    assert any("2/4" in m and "50.0%" in m for m in messages)
    assert any("4/4" in m and "100.0%" in m for m in messages)


def test_stage_throttles_advance_logging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    clock = _FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    interval = 5.0
    with stage(Progress(min_log_interval=interval), "X", total=100) as s:
        caplog.clear()
        s.advance()
        s.advance()
        s.advance()
        assert caplog.records == []

        clock.advance(interval)
        s.advance()
        assert len(caplog.records) == 1


def test_stage_exit_logs_on_exception(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    with pytest.raises(ValueError, match="boom"):
        with stage(Progress(), "X"):
            raise ValueError("boom")

    messages = [r.message for r in caplog.records]
    assert any("X: failed" in m for m in messages)


def test_none_progress_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    with stage(None, "X") as s:
        s.advance()
        s.advance(5)

    assert caplog.records == []


def test_heartbeat_logs_periodically_for_unknown_total(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    with stage(Progress(min_log_interval=0.02), "X"):
        time.sleep(0.06)

    messages = [r.message for r in caplog.records]
    assert any(re.fullmatch(r"X \d+\.\ds", m) for m in messages)


def test_advance_call_count_matches_loop_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = ProgressStage.advance

    def counting_advance(self: ProgressStage, n: int = 1) -> None:
        calls.append(n)
        original(self, n)

    monkeypatch.setattr(ProgressStage, "advance", counting_advance)

    with stage(Progress(), "X", total=3) as s:
        for _ in range(3):
            s.advance()

    assert calls == [1, 1, 1]


def test_format_duration_across_magnitudes() -> None:
    assert _format_duration(9.1) == "9.1s"
    assert _format_duration(128.4) == "2m8s"
    assert _format_duration(7384.2) == "2h3m"


def test_nested_stage_does_not_log_its_own_started_or_done(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    progress = Progress(min_log_interval=0.0)

    with stage(progress, "Outer"):
        with stage(progress, "Inner"):
            pass

    messages = [r.message for r in caplog.records]
    assert any("Outer: started" in m for m in messages)
    assert any("Outer: done" in m for m in messages)
    assert not any("Inner: started" in m for m in messages)
    assert not any("Inner: done" in m for m in messages)


def test_breadcrumb_contains_all_open_frames_innermost_last(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    progress = Progress(min_log_interval=0.0)

    with stage(progress, "Outer", total=2):
        with stage(progress, "Inner", total=4) as inner:
            inner.advance()

    breadcrumbs = [
        r.message
        for r in caplog.records
        if "Outer" in r.message and "Inner" in r.message
    ]
    assert breadcrumbs
    line = breadcrumbs[-1]
    assert line.index("Outer") < line.index("Inner")


def test_inner_advance_renders_even_when_outer_is_static(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    progress = Progress(min_log_interval=0.0)

    with stage(progress, "Outer", total=2):
        caplog.clear()
        with stage(progress, "Inner", total=4) as inner:
            inner.advance()

    messages = [r.message for r in caplog.records]
    assert any("Inner 1/4" in m for m in messages)


def test_throttle_is_shared_across_nesting_levels(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    clock = _FakeClock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)

    interval = 5.0
    progress = Progress(min_log_interval=interval)

    with stage(progress, "Outer", total=10) as outer:
        caplog.clear()
        outer.advance()
        with stage(progress, "Inner", total=10) as inner:
            inner.advance()
        assert caplog.records == []

        clock.advance(interval)
        outer.advance()
        assert len(caplog.records) == 1


def test_render_happens_before_pop_on_stage_exit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    progress = Progress(min_log_interval=0.0)

    with stage(progress, "Outer", total=2):
        with stage(progress, "Inner", total=4):
            pass

    # The line rendered as "Inner" exits (before it's popped) is the only
    # place its completion, with a final duration, is visible at all.
    assert any("Inner" in r.message for r in caplog.records)


def test_nested_stages_never_move_an_enclosing_counter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # advance() moves only the stage it is called on: opening, closing or
    # advancing a child leaves every enclosing counter alone.
    caplog.set_level(logging.INFO)
    progress = Progress(min_log_interval=0.0)

    with stage(progress, "Outer", total=2):
        with stage(progress, "Inner", total=2) as inner:
            inner.advance()
            inner.advance()
        with stage(progress, "Inner", total=1):
            pass

    messages = [r.message for r in caplog.records]
    assert any("Outer 0/2" in m for m in messages)
    assert not any("Outer 1/2" in m for m in messages)
    assert not any("Outer 2/2" in m for m in messages)
