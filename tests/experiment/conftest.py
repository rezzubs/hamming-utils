"""Shared test doubles for the faultforge.experiment tests."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from faultforge import Fingerprint
from faultforge.experiment import Experiment, StopCondition
from pydantic import BaseModel

# The following classes are `_` prefixed to not interpret them as Test classes.


def _fingerprint(name: str = "test") -> Fingerprint:
    return Fingerprint(kind="test", scalars={"name": name})


@dataclass(slots=True)
class _TestResult:
    value: float


class _SavedData(BaseModel):
    fingerprint: Fingerprint
    results: dict[int, float]


class _TestExperiment(Experiment):
    _fingerprint: Fingerprint
    _results: dict[int, _TestResult]
    _intrinsic_stop_conditions: list[StopCondition]

    def __init__(
        self,
        results: dict[int, _TestResult] | None = None,
        name: str = "test",
    ) -> None:
        self._fingerprint = _fingerprint(name)
        self._results = results or {}
        self._intrinsic_stop_conditions = []

    def add_stop_condition(self, condition: StopCondition) -> None:
        """Contribute an additional intrinsic condition, as if a subclass had
        overridden `stop_conditions` itself."""
        self._intrinsic_stop_conditions.append(condition)

    @override
    def stop_conditions(self) -> Sequence[StopCondition]:
        return self._intrinsic_stop_conditions

    @override
    def scores(self) -> Sequence[float]:
        return [result.value for result in self._results.values()]

    @override
    def run(self) -> None:
        key = len(self._results)
        self._results[key] = _TestResult(value=float(key + 1))

    @override
    def serialize(self) -> str:
        return _SavedData(
            fingerprint=self._fingerprint,
            results={key: result.value for key, result in self._results.items()},
        ).model_dump_json()

    @override
    def deserialize(self, content: str) -> None:
        loaded = _SavedData.model_validate_json(content)
        self._fingerprint.raise_if_differs(loaded.fingerprint)
        self._results = {
            key: _TestResult(value=value) for key, value in loaded.results.items()
        }


def make(values: list[float] | None = None, name: str = "test") -> _TestExperiment:
    """Create a test experiment with existing results"""
    results = {key: _TestResult(value=value) for key, value in enumerate(values or [])}
    return _TestExperiment(results=results, name=name)
