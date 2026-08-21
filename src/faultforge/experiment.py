"""Classes for running experiments.

See `Experiment` for the full overview.
"""

from faultforge._internal.experiment import (
    AdditionalRuns,
    Experiment,
    ExperimentDisplay,
    MaxRuns,
    SaveConfig,
    Stability,
    StopCondition,
    relative_margin_of_error,
)

__all__ = [
    "AdditionalRuns",
    "Experiment",
    "ExperimentDisplay",
    "MaxRuns",
    "SaveConfig",
    "Stability",
    "StopCondition",
    "relative_margin_of_error",
]
