"""Tests for ExperimentDisplay.extra."""

from typing import override

from faultforge.experiment import ExperimentDisplay, Stability

# `_` prefixed to not interpret it as a Test class.


class _DisplayWithExtra(ExperimentDisplay):
    def __init__(self, value: str | None) -> None:
        self._value = value

    @override
    def extra(self) -> str | None:
        return self._value


def test_extra_appended_as_is():
    # `extra` is appended verbatim, with no separator added by `format` - the
    # override is responsible for its own leading spacing/delimiter.
    display = _DisplayWithExtra(" | note")
    status = display.format(
        run_count=1,
        score=1.0,
        mean=1.5,
        margin_of_error=0.5,
        stop_conditions=[Stability(min_samples=0, threshold=1.0)],
    )
    assert status.endswith(" | note")


def test_extra_omitted_when_none():
    display = _DisplayWithExtra(None)
    status = display.format(
        run_count=1,
        score=1.0,
        mean=1.5,
        margin_of_error=0.5,
        stop_conditions=[Stability(min_samples=0, threshold=1.0)],
    )
    assert "note" not in status


def test_extra_shown_without_stability_condition():
    # `extra` applies regardless of whether the `Relative MoE` fragment ends
    # up in the line, since a `Stability` condition need not be configured.
    display = _DisplayWithExtra(" | note")
    status = display.format(
        run_count=1,
        score=1.0,
        mean=1.5,
        margin_of_error=0.5,
        stop_conditions=[],
    )
    assert status.endswith(" | note")


def test_extra_shown_with_single_score():
    # `extra` applies even when there's no mean/margin_of_error yet.
    display = _DisplayWithExtra(" | note")
    status = display.format(
        run_count=1,
        score=1.0,
        mean=None,
        margin_of_error=None,
    )
    assert status.endswith(" | note")
