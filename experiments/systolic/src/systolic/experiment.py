"""An experiment for measuring model reliability under systolic-array register faults.

See `systolic` for a general overview.
"""

import copy
import enum
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, final, override

import torch
from faultforge import Fingerprint, Picker
from faultforge.dataset import DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, DeviceLike
from faultforge.experiment import Experiment, ExperimentDisplay, StopCondition
from faultforge.io import AnyPath, is_compressed, open_text
from faultforge.loading import DEFAULT_DTYPE, ModelBundle
from faultforge.progress import Progress, stage
from pydantic import BaseModel
from torch import Tensor

from systolic._rust import ArrayConfig, Fault
from systolic.backend import SystolicBackend
from systolic.fault import RegisterFaults
from systolic.lifted_backend import LiftedBackend
from systolic.model import BackendModel
from systolic.simulated_backend import SimulatedBackend

logger = logging.getLogger(__name__)

BackendKind = Literal["simulated", "lifted"]


class ReliabilityMetric(enum.StrEnum):
    """Ways to measure the reliability of a fault-injected model."""

    Accuracy = "accuracy"
    """Correct predictions / total predictions.

    Whether or not a prediction is "correct" is defined by the dataset's
    ground-truth labels (targets).
    """
    AccuracyDegradation = "accuracy_degradation"
    """Golden model accuracy - faulty model accuracy."""
    Sdc = "sdc"
    """Silent Data Corruption. Change in any output logit vs the golden model."""
    Top1Sdc = "top1_sdc"
    """Critical Silent Data Corruption. Change in top-1 logit (the prediction) vs the golden model."""

    def requires_golden(self) -> bool:
        """Whether this metric also requires evaluating results on a golden model."""

        return self.value in {
            ReliabilityMetric.AccuracyDegradation.value,
            ReliabilityMetric.Sdc.value,
            ReliabilityMetric.Top1Sdc.value,
        }

    def score_name(self) -> str:
        """The name of the score for this metric."""
        match self:
            case ReliabilityMetric.Accuracy:
                return "Accuracy"
            case ReliabilityMetric.AccuracyDegradation:
                return "Accuracy Degradation"
            case ReliabilityMetric.Sdc:
                return "SDC"
            case ReliabilityMetric.Top1Sdc:
                return "Top-1 SDC"


def compute_score(metric: ReliabilityMetric, correct: int, total: int) -> float:
    """Score a single run's correct/total accounting under `metric`."""
    match metric:
        case ReliabilityMetric.Sdc | ReliabilityMetric.Top1Sdc:
            return 100 - float(correct) / float(total) * 100
        case ReliabilityMetric.Accuracy | ReliabilityMetric.AccuracyDegradation:
            return float(correct) / float(total) * 100


@final
@dataclass(slots=True)
class BatchReliability:
    correct: int
    """Number of correct results as defined by the metric."""
    total: int
    """Total number of "items" in the batch. Metric dependent."""

    def __add__(self, other: BatchReliability) -> BatchReliability:
        return BatchReliability(
            correct=self.correct + other.correct, total=self.total + other.total
        )


def _batch_critical_sdc(
    logits: Tensor, golden_classifications: Tensor
) -> BatchReliability:
    """Compute the critical SDC of a result. Used for ReliabilityMetric.Top1Sdc."""

    classifications = logits.argmax(dim=1)

    assert golden_classifications.shape == classifications.shape
    correct = int((classifications == golden_classifications).sum().item())
    total = golden_classifications.numel()

    return BatchReliability(correct=correct, total=total)


def _batch_sdc(logits: Tensor, golden_logits: Tensor) -> BatchReliability:
    """Compute the SDC of a result. Used for ReliabilityMetric.Sdc."""

    assert golden_logits.shape == logits.shape
    correct = int((logits == golden_logits).sum().item())
    total = golden_logits.numel()

    return BatchReliability(correct=correct, total=total)


def _batch_accuracy_degradation(
    logits: Tensor,
    golden_classifications: Tensor,
    targets: Tensor,
) -> BatchReliability:
    """Compute the accuracy degradation of a result. Used for ReliabilityMetric.AccuracyDegradation."""

    classifications = logits.argmax(dim=1)
    assert golden_classifications.shape == classifications.shape

    correct = int((classifications == targets).sum().item())
    golden_correct = int((golden_classifications == targets).sum().item())
    total = classifications.numel()

    return BatchReliability(correct=golden_correct - correct, total=total)


def _batch_accuracy(logits: Tensor, targets: Tensor) -> BatchReliability:
    """Compute the accuracy of a result. Used for ReliabilityMetric.Accuracy."""
    classifications = logits.argmax(dim=1)

    correct = int((classifications == targets).sum().item())
    total = classifications.numel()

    return BatchReliability(correct=correct, total=total)


class SavedResult(BaseModel):
    """The on-disk shape of a `SystolicFaultInjection`'s results.

    Standalone-loadable: everything needed to recompute scores and resume
    the fault `Picker` lives here, so a result file can be inspected without
    reconstructing the model/dataset that produced it.
    """

    fingerprint: Fingerprint
    total_items: int | None
    correct_counts: list[int]
    """One entry per run, in run order."""
    fault_ids: list[int]
    """The fault id drawn for each run, in the same order as `correct_counts`.
    Feeds `Picker.from_returned` on resume."""

    @classmethod
    def load(cls, path: AnyPath) -> SavedResult:
        """Load a single result file previously written by
        `Experiment.save`/`save_atomic`."""
        with open_text(path, "rt", compressed=is_compressed(path)) as f:
            return cls.model_validate_json(f.read())

    def reliability_metric(self) -> ReliabilityMetric:
        return ReliabilityMetric(self.fingerprint.scalars["reliability_metric"])

    def scores(self) -> list[float]:
        """Every recorded run's score, in run order.

        Empty if `total_items` is `None` (no run has completed yet).
        """
        if self.total_items is None:
            return []
        metric = self.reliability_metric()
        return [
            compute_score(metric, correct, self.total_items)
            for correct in self.correct_counts
        ]


class _Display(ExperimentDisplay):
    """`SystolicFaultInjection`'s display: names/units the score per metric."""

    def __init__(self, metric: ReliabilityMetric) -> None:
        self._metric = metric

    @override
    def score_name(self) -> str | None:
        return self._metric.score_name()

    @override
    def score_unit(self) -> str | None:
        return "%"


@final
class SystolicFaultInjection(Experiment):
    """An experiment which injects a single stuck-at register fault per run
    into a weight-stationary systolic array evaluating the model.

    One fault is active per run (not several simultaneous faults, unlike
    `encoded_memory`'s `EncodedFaultInjection`): each run draws one fault id
    from a `Picker` over `fault.radix(array)`, applies it via
    `backend.set_fault`, and scores one inference pass over the dataset. How
    many runs happen is controlled entirely by stop conditions.
    """

    def __init__(
        self,
        bundle: ModelBundle,
        array: tuple[int, int],
        *,
        fault: RegisterFaults = RegisterFaults(),
        backend: BackendKind = "lifted",
        reliability_metric: ReliabilityMetric = ReliabilityMetric.Accuracy,
        preload_dataset: bool = True,
        dataset_batch_limit: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: DeviceLike = DEFAULT_DEVICE,
        dtype: torch.dtype = DEFAULT_DTYPE,
        progress: Progress | None = None,
    ) -> None:
        if dtype != torch.float32:
            raise ValueError(
                f"SystolicFaultInjection only supports float32, got {dtype}"
            )

        self._progress = progress
        self._golden_results: list[Tensor] = []
        self._total_items: int | None = None
        self._correct_counts: list[int] = []
        self._fault_ids: list[int] = []
        self._reliability_metric = reliability_metric
        self._fault_config = fault
        self._dtype = dtype

        nrows, ncols = array
        self._array = ArrayConfig(nrows, ncols, 32)

        model = bundle.load_model(device, dtype=dtype, progress=progress)
        self._golden_model = copy.deepcopy(model)

        backend_instance: SystolicBackend
        match backend:
            case "simulated":
                backend_instance = SimulatedBackend(nrows, ncols)
            case "lifted":
                backend_instance = LiftedBackend(nrows, ncols)
        self._backend = backend_instance
        self._model = BackendModel(model, self._backend, progress=progress)

        self._dataset = bundle.load_dataset(batch_size, device, progress=progress)
        if dataset_batch_limit is not None and not preload_dataset:
            logger.warning(
                "preload_dataset is set to False but dataset_limit forces a preload anyway"
            )
            preload_dataset = True
        if preload_dataset:
            self._dataset = self._dataset.precompute(
                dataset_batch_limit, progress=progress
            )

        fingerprint = Fingerprint(
            kind="systolic_fault_injection",
            scalars={
                "reliability_metric": reliability_metric.value,
                "backend": backend,
                "array_nrows": nrows,
                "array_ncols": ncols,
                "registers": fault.fingerprint_scalar(),
                "dtype": "f32",
            },
            children={"bundle": [bundle.fingerprint()]},
        )
        test_image_limit = (
            dataset_batch_limit * batch_size
            if dataset_batch_limit is not None
            else None
        )
        if test_image_limit is not None:
            fingerprint.scalars["test_image_limit"] = test_image_limit
        self._fingerprint = fingerprint

        self._picker = Picker(self._fault_config.radix(self._array))

    def _process_golden(self, golden_result: Tensor) -> Tensor:
        match self._reliability_metric:
            case (
                ReliabilityMetric.Top1Sdc
                | ReliabilityMetric.Accuracy
                | ReliabilityMetric.AccuracyDegradation
            ):
                return golden_result.argmax(dim=1)
            case ReliabilityMetric.Sdc:
                return golden_result

    def _populate_golden(self) -> None:
        """Populate the golden results and set `_total_items`."""
        total_items = 0

        try:
            with (
                stage(
                    self._progress,
                    "Computing golden results",
                    total=self._dataset.batch_count(),
                ) as s,
                torch.no_grad(),
            ):
                for batch in self._dataset:
                    logits = self._golden_model.forward(
                        batch.inputs.to(dtype=self._dtype)
                    )
                    processed = self._process_golden(logits)
                    total_items += processed.numel()
                    self._golden_results.append(processed)
                    s.advance()
        finally:
            self._dataset.reset()

        if self._total_items is None:
            self._total_items = total_items
        else:
            assert self._total_items == total_items, (
                "_total_items mismatch vs previous run"
            )

    def _score(self, correct: int) -> float:
        if self._total_items is None:
            raise RuntimeError("Unable to score a result before the first run")
        return compute_score(self._reliability_metric, correct, self._total_items)

    @override
    def scores(self) -> Sequence[float]:
        if self._total_items is None:
            return []
        return [self._score(correct) for correct in self._correct_counts]

    @override
    def display(self) -> ExperimentDisplay:
        return _Display(self._reliability_metric)

    @override
    def stop_conditions(self) -> Sequence[StopCondition]:
        def picker_exhausted(experiment: Experiment) -> str | None:
            _ = experiment
            if self._picker.size == 0:
                return "Picker exhausted: every fault in the (restricted) fault space has been sampled"
            return None

        return (picker_exhausted,)

    @override
    def serialize(self) -> str:
        return SavedResult(
            fingerprint=self._fingerprint,
            total_items=self._total_items,
            correct_counts=self._correct_counts,
            fault_ids=self._fault_ids,
        ).model_dump_json()

    @override
    def deserialize(self, content: str) -> None:
        loaded = SavedResult.model_validate_json(content)
        self._fingerprint.raise_if_differs(loaded.fingerprint)
        self._total_items = loaded.total_items
        self._correct_counts = loaded.correct_counts
        self._fault_ids = loaded.fault_ids
        self._picker = Picker.from_returned(
            self._fault_config.radix(self._array), set(self._fault_ids)
        )

    def _draw_fault(self) -> tuple[int, Fault]:
        try:
            fault_id = next(self._picker)
        except StopIteration:
            raise RuntimeError(
                "Expected a fault id to be available but the picker is exhausted"
            ) from None
        return fault_id, self._fault_config.fault_from_id(fault_id, self._array)

    def _infer(self) -> BatchReliability:
        """Run inference on `self._model` over the dataset, scored by `self._reliability_metric`."""
        result = BatchReliability(correct=0, total=0)
        with (
            stage(self._progress, "Inference", total=self._dataset.batch_count()) as s,
            torch.no_grad(),
        ):
            for batch_index, batch in enumerate(self._dataset):
                logits = self._model.forward(batch.inputs.to(dtype=self._dtype))

                match self._reliability_metric:
                    case ReliabilityMetric.Accuracy:
                        batch_result = _batch_accuracy(logits, batch.targets)
                    case ReliabilityMetric.AccuracyDegradation:
                        batch_result = _batch_accuracy_degradation(
                            logits, self._golden_results[batch_index], batch.targets
                        )
                    case ReliabilityMetric.Sdc:
                        batch_result = _batch_sdc(
                            logits, self._golden_results[batch_index]
                        )
                    case ReliabilityMetric.Top1Sdc:
                        batch_result = _batch_critical_sdc(
                            logits, self._golden_results[batch_index]
                        )

                result += batch_result
                s.advance()

        self._dataset.reset()
        return result

    def _record_result(self, fault_id: int, result: BatchReliability) -> None:
        if self._total_items is None:
            self._total_items = result.total
            assert not self._reliability_metric.requires_golden(), (
                "_total_items should be set by _populate_golden"
            )

        if result.total != self._total_items:
            raise RuntimeError(
                f"Computed {self._total_items} elements from the golden results, "
                f"model returned {result.total}"
            )

        self._fault_ids.append(fault_id)
        self._correct_counts.append(result.correct)

    @override
    def run(self) -> None:
        if not self._golden_results and self._reliability_metric.requires_golden():
            self._populate_golden()

        fault_id, fault = self._draw_fault()
        self._backend.set_fault(fault)

        result = self._infer()
        self._record_result(fault_id, result)
