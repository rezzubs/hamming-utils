"""An experiment for measuring model reliability under systolic-array register faults.

`SystolicFaultInjection` wraps a model with a `faultforge.systolic.SystolicBackend`
and, per run, injects a single stuck-at fault into one register of one
processing element, scoring the result according to a `ReliabilityMetric`.
"""

from faultforge._internal.experiments.systolic import (
    BackendKind,
    ReliabilityMetric,
    SavedResult,
    SystolicFaultInjection,
)

__all__ = [
    "BackendKind",
    "ReliabilityMetric",
    "SavedResult",
    "SystolicFaultInjection",
]
