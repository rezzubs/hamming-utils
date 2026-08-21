"""An experiment for measuring model reliability under memory faults.

See `encoded_memory` for a general overview.
"""

import copy
import enum
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, final, override

import torch
from faultforge import BitFlip, Fingerprint, Picker, bitwise_xor
from faultforge.dataset import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    BatchedDataset,
    DeviceLike,
)
from faultforge.dtype import EncodingDtype, FiDtype
from faultforge.encoding import EncodedModule, Encoder
from faultforge.experiment import Experiment, ExperimentDisplay
from faultforge.io import AnyPath, is_compressed, open_text
from faultforge.loading import DEFAULT_DTYPE, ModelBundle
from faultforge.progress import Progress, stage
from pydantic import BaseModel, Field
from torch import Tensor, nn

logger = logging.getLogger(__name__)


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
    """Score a single run's correct/total accounting under `metric`.

    Pure so it can be reused both by a live `EncodedFaultInjection` (via
    `_score`) and by `SavedResult.scores`, which recomputes scores from a
    saved file without reconstructing a model/dataset.
    """
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


class SimpleResult(BaseModel):
    """Correct/total accounting only."""

    kind: Literal["simple"] = "simple"
    results: list[int]

    def correct_counts(self) -> list[int]:
        return self.results


class DetailedRunResult(BaseModel):
    """A single run's correct/total accounting plus its bitwise-comparison data."""

    correct_count: int
    bitmask: list[int]
    """Flat list of nonzero xor values between the faulty and golden parameters,
    across all parameter tensors."""


class DetailedResult(BaseModel):
    """Correct/total accounting plus per-run bitwise-comparison data.

    Each run contributes a single `DetailedRunResult`, so `correct` and
    `bitmask` can never drift out of sync the way two parallel lists could.
    """

    kind: Literal["detailed"] = "detailed"
    results: list[DetailedRunResult]

    def correct_counts(self) -> list[int]:
        return [run.correct_count for run in self.results]

    def discard_bitmasks(self) -> SimpleResult:
        """Drop the recorded bitmasks, keeping only the correct/total accounting."""
        return SimpleResult(results=self.correct_counts())


ExperimentResult = Annotated[SimpleResult | DetailedResult, Field(discriminator="kind")]


class SavedResult(BaseModel):
    """The on-disk shape of an `EncodedFaultInjection`'s results.

    Standalone-loadable: everything needed to recompute scores and bit
    error rate lives here, so a result file can be inspected (e.g. for
    plotting) without reconstructing the model/dataset that produced it.
    """

    fingerprint: Fingerprint
    total_items: int | None
    total_bits: int
    """The size (in bits) of the encoded memory faults were injected into.

    Recorded directly rather than left as part of `Fingerprint.scalars`,
    since it's already fully determined by the model/encoder/dtype
    fingerprints that *are* part of identity - this just avoids recomputing
    it, the same rationale as `total_items`.
    """
    result: ExperimentResult

    @classmethod
    def load(cls, path: AnyPath) -> SavedResult:
        """Load a single result file previously written by
        `Experiment.save`/`save_atomic`.

        Unlike reconstructing an `EncodedFaultInjection`, this doesn't
        require the model/dataset that produced it - just the file itself.
        """
        path = Path(path).expanduser()
        with open_text(path, "rt", compressed=is_compressed(path)) as f:
            return cls.model_validate_json(f.read())

    def reliability_metric(self) -> ReliabilityMetric:
        return ReliabilityMetric(self.fingerprint.scalars["reliability_metric"])

    def scores(self) -> list[float]:
        """Every recorded run's score, in run order.

        Empty if `total_items` is `None` (no run has completed yet),
        mirroring `EncodedFaultInjection.scores()`.
        """
        if self.total_items is None:
            return []
        metric = self.reliability_metric()
        return [
            compute_score(metric, correct, self.total_items)
            for correct in self.result.correct_counts()
        ]

    def bit_error_rate(self) -> float:
        """The realized fraction of encoded bits flipped, `faults / total_bits`."""
        faults = self.fingerprint.scalars["faults"]
        assert isinstance(faults, int)
        return faults / self.total_bits


def _discard_bitmasks(
    result: SimpleResult | DetailedResult, fingerprint: Fingerprint
) -> tuple[SimpleResult | DetailedResult, Fingerprint]:
    """Drop any recorded bitmasks, converting to the simpler result kind.

    Also flips the fingerprint's `compare_bitwise` scalar to `False`, since
    otherwise comparing this fingerprint against one from a freshly
    constructed `EncodedFaultInjection(..., compare_bitwise=False)` would
    report a spurious mismatch even though the result kinds now agree. A
    no-op if bitmasks weren't being recorded in the first place.
    """
    if not isinstance(result, DetailedResult):
        return result, fingerprint

    updated_fingerprint = fingerprint.model_copy(
        update={"scalars": {**fingerprint.scalars, "compare_bitwise": False}}
    )
    return result.discard_bitmasks(), updated_fingerprint


def discard_bitmasks_in_file(path: AnyPath) -> None:
    """Discard any recorded bitmasks from a saved `EncodedFaultInjection` result file.

    Unlike `EncodedFaultInjection.discard_bitmasks`, this reads and rewrites a
    file previously written by `Experiment.save`/`save_atomic` directly, so it
    doesn't require reconstructing the model/dataset that produced it. Writes
    back atomically, the same way `Experiment.save_atomic` does. Whichever
    format (zstd-compressed or not) `path` was already in is preserved on
    write-back. A no-op (besides rewriting the file) if bitmasks weren't
    recorded in the first place.
    """
    path = Path(path).expanduser()
    compressed = is_compressed(path)
    loaded = SavedResult.load(path)
    result, fingerprint = _discard_bitmasks(loaded.result, loaded.fingerprint)
    updated = SavedResult(
        fingerprint=fingerprint,
        total_items=loaded.total_items,
        total_bits=loaded.total_bits,
        result=result,
    ).model_dump_json()

    fd, temp_name = tempfile.mkstemp()
    os.close(fd)
    with open_text(temp_name, "wt", compressed=compressed) as temp:
        temp.write(updated)
    os.replace(temp_name, path)


def _bit_histogram(bitmask: Sequence[int]) -> dict[int, int]:
    """Count how many `bitmask` elements have each number of set bits."""
    histogram: dict[int, int] = {}
    for value in bitmask:
        ones = value.bit_count()
        histogram[ones] = histogram.get(ones, 0) + 1
    return histogram


@dataclass(slots=True, frozen=True)
class _FaultInjectionSummary:
    """A single run's fault-injection stats, for display via `_Display.extra`.

    Not part of any serialized result - purely a display-time snapshot of the
    latest run.
    """

    faults_injected: int
    total_bits: int
    bit_histogram: dict[int, int] | None
    """Maps "faulty bits in one element" -> "how many elements had that many".
    `None` when bitwise comparison wasn't recorded for this run
    (`compare_bitwise=False`)."""

    def bit_error_rate(self) -> float:
        return self.faults_injected / self.total_bits

    @override
    def __str__(self) -> str:
        lines = [
            f"Flipped {self.faults_injected}/{self.total_bits} bits "
            f"- BER: {self.bit_error_rate():.2e}"
        ]

        if self.bit_histogram is not None:
            if self.faults_injected > 0:
                measured = sum(
                    bits * count for bits, count in self.bit_histogram.items()
                )
                affected = sum(self.bit_histogram.values())
                masked = (1 - measured / self.faults_injected) * 100
                lines.append(f"{affected} parameters were affected")
                lines.append(
                    f"{measured} bits were measured faulty ({masked:.2f}% masked)"
                )
            for bits, count in sorted(self.bit_histogram.items()):
                plural = "s" if bits != 1 else ""
                lines.append(f"{count} parameters had {bits} faulty bit{plural}")

        return "\n".join(lines)


class _Display(ExperimentDisplay):
    """`EncodedFaultInjection`'s display: names/units the score per metric."""

    def __init__(
        self, metric: ReliabilityMetric, fault_summary: _FaultInjectionSummary | None
    ) -> None:
        self._metric = metric
        self._fault_summary = fault_summary

    @override
    def score_name(self) -> str | None:
        return self._metric.score_name()

    @override
    def score_unit(self) -> str | None:
        return "%"

    @override
    def extra(self) -> str | None:
        if self._fault_summary is None:
            return None
        return "\n" + str(self._fault_summary)


@final
class EncodedFaultInjection(Experiment):
    """An experiment which emulates single-event upsets in the encoded memory that stores model parameters."""

    _model: EncodedModule
    _dataset: BatchedDataset
    _device: torch.device
    _dtype: torch.dtype
    _reliability_metric: ReliabilityMetric
    _faulty_bit_count: int
    _total_bits: int
    _progress: Progress | None
    _fingerprint: Fingerprint
    _show_fault_summary: bool

    _unencoded_golden: nn.Module | None

    # populated during first run
    _golden_results: list[Tensor]
    _total_items: int | None
    _result: SimpleResult | DetailedResult
    _last_fault_summary: _FaultInjectionSummary | None

    def __init__(
        self,
        bundle: ModelBundle,
        encoder: Encoder,
        reliability_metric: ReliabilityMetric,
        *,
        golden_is_encoded: bool = False,
        faults: int | float = 1,
        compare_bitwise: bool = False,
        fault_summary: bool = False,
        preload_dataset: bool = True,
        dataset_batch_limit: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: DeviceLike = DEFAULT_DEVICE,
        dtype: torch.dtype = DEFAULT_DTYPE,
        progress: Progress | None = None,
    ) -> None:
        self._progress = progress
        self._golden_results = []
        self._total_items = None
        self._result = (
            DetailedResult(results=[]) if compare_bitwise else SimpleResult(results=[])
        )
        self._show_fault_summary = fault_summary
        self._last_fault_summary = None

        model = bundle.load_model(device, dtype=dtype, progress=progress)
        if golden_is_encoded:
            self._unencoded_golden = None
        else:
            self._unencoded_golden = copy.deepcopy(model)

        self._model = EncodedModule(model, encoder, progress=progress)
        self._device = torch.device(device)
        self._dtype = dtype
        self._reliability_metric = reliability_metric

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
            kind="encoded_memory_fault_injection",
            scalars={
                "reliability_metric": reliability_metric.value,
                "golden": "encoded" if golden_is_encoded else "unencoded",
                "compare_bitwise": compare_bitwise,
                "dtype": EncodingDtype.from_torch(dtype).value,
            },
            children={
                "bundle": [bundle.fingerprint()],
                "encoder": [encoder.fingerprint()],
            },
        )

        test_image_limit = (
            dataset_batch_limit * batch_size
            if dataset_batch_limit is not None
            else None
        )
        if test_image_limit is not None:
            fingerprint.scalars["test_image_limit"] = test_image_limit

        self._total_bits = self._model.bit_count()

        if isinstance(faults, int):
            if faults > self._total_bits:
                raise ValueError(
                    f"`faults` ({faults}) is greater than the number of bits in model parameters ({self._total_bits})"
                )

            self._faulty_bit_count = faults
        elif isinstance(faults, float):
            if faults > 1.0:
                raise ValueError(
                    f"`faults` ({faults}) is greater than 1.0 (floats are interpreted as the bit error rate)"
                )

            self._faulty_bit_count = int(round(faults * self._total_bits))
            logger.debug(
                f"Resolved bit error rate {faults} to {self._faulty_bit_count} faults"
            )

        # Always store the resolved fault count rather than branching on
        # which of `faults`/`bit_error_rate` the caller passed in, so that
        # e.g. `faults=328` and a `bit_error_rate` that happens to resolve to
        # 328 for this exact model produce identical fingerprints (otherwise
        # resuming a saved file with the other input style would spuriously
        # fail the fingerprint check).
        fingerprint.scalars["faults"] = self._faulty_bit_count

        self._fingerprint = fingerprint

    def _process_golden(self, golden_result: Tensor) -> Tensor:
        """Run a function on the golden result after computing it.

        This enables processing the results only once. The result will be given to
        the batch reliability functions.
        """
        match self._reliability_metric:
            case (
                ReliabilityMetric.Top1Sdc
                | ReliabilityMetric.Accuracy
                | ReliabilityMetric.AccuracyDegradation
            ):
                return golden_result.argmax(dim=1)
            case ReliabilityMetric.Sdc:
                return golden_result

    def _populate_golden(self):
        """Populate the golden results.

        Additionally sets `_total_items` to the total number of predictions;
        this is used for computing SDC scores as well as the number of
        injected faults.
        """
        total_items = 0

        golden: nn.Module = self._unencoded_golden or self._model

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
                    logits = golden.forward(batch.inputs.to(dtype=self._dtype))
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
        return [self._score(correct) for correct in self._result.correct_counts()]

    @override
    def display(self) -> ExperimentDisplay:
        return _Display(self._reliability_metric, self._last_fault_summary)

    def discard_bitmasks(self) -> None:
        """Drop any recorded bitmasks, converting to the simpler result kind.

        A no-op if bitmasks weren't being recorded in the first place.
        """
        self._result, self._fingerprint = _discard_bitmasks(
            self._result, self._fingerprint
        )

    @override
    def serialize(self) -> str:
        return SavedResult(
            fingerprint=self._fingerprint,
            total_items=self._total_items,
            total_bits=self._total_bits,
            result=self._result,
        ).model_dump_json()

    @override
    def deserialize(self, content: str) -> None:
        loaded = SavedResult.model_validate_json(content)
        self._fingerprint.raise_if_differs(loaded.fingerprint)
        self._total_items = loaded.total_items
        self._total_bits = loaded.total_bits
        self._result = loaded.result

    def _inject_faults(self) -> EncodedModule:
        """Clone the model and flip `self._faulty_bit_count` unique random bits in it."""
        picker = Picker(self._model.bit_count())
        model = self._model.clone()
        with stage(self._progress, "Fault Injection"):
            fault_targets: list[tuple[BitFlip, int]] = []
            for _ in range(self._faulty_bit_count):
                try:
                    fault_target = next(picker)
                except StopIteration:
                    raise RuntimeError(
                        "Expected fault targets to be within range but picker is exhausted"
                    )
                fault_targets.append((BitFlip(), fault_target))

            model.apply_faults(fault_targets)
        return model

    def _compare_bitwise(self, model: EncodedModule) -> list[int] | None:
        """Bitwise-compare `model`'s decoded parameters against the golden ones.

        Returns the flat list of nonzero (unsigned) xor values across all
        parameter tensors, or `None` when `compare_bitwise=False` (i.e.
        `self._result` isn't a `DetailedResult`).
        """
        if not isinstance(self._result, DetailedResult):
            return None

        faulty_params = list(model.decode().parameters())
        if self._unencoded_golden is not None:
            golden_params = list(self._unencoded_golden.parameters())
        else:
            golden_params = list(self._model.decode().parameters())

        # `xor` is a bitcast view of a signed dtype (see `bitwise_xor`), so
        # e.g. an all-ones 32-bit pattern comes back as `-1`. Masking to the
        # dtype's bit width recovers the true unsigned bit pattern, relying
        # on Python's arbitrary-precision two's-complement semantics
        # (`-1 & 0xFFFFFFFF == 0xFFFFFFFF`).
        mask = (1 << FiDtype.from_torch(self._dtype).bit_width()) - 1

        with stage(self._progress, "Bitwise Comparison", total=len(golden_params)) as s:
            bitmask: list[int] = []
            for faulty, golden in zip(faulty_params, golden_params, strict=True):
                xor = bitwise_xor(faulty, golden)
                bitmask.extend(value & mask for value in xor[xor != 0].tolist())
                s.advance()

        return bitmask

    def _infer(self, model: EncodedModule) -> BatchReliability:
        """Run inference on `model` over the dataset, scored by `self._reliability_metric`."""
        result = BatchReliability(correct=0, total=0)
        with (
            stage(self._progress, "Inference", total=self._dataset.batch_count()) as s,
            torch.no_grad(),
        ):
            for batch_index, batch in enumerate(self._dataset):
                # n_batches x n_classes
                logits = model.forward(batch.inputs.to(dtype=self._dtype))

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

    def _record_result(
        self, result: BatchReliability, bitmask: list[int] | None
    ) -> None:
        """Validate `result`'s totals, then append it (and `bitmask`) to `self._result`."""
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

        if isinstance(self._result, DetailedResult):
            assert bitmask is not None
            self._result.results.append(
                DetailedRunResult(correct_count=result.correct, bitmask=bitmask)
            )
        else:
            self._result.results.append(result.correct)

        if self._show_fault_summary:
            self._last_fault_summary = _FaultInjectionSummary(
                faults_injected=self._faulty_bit_count,
                total_bits=self._model.bit_count(),
                bit_histogram=(
                    _bit_histogram(bitmask) if bitmask is not None else None
                ),
            )

    @override
    def run(self) -> None:
        if not self._golden_results and self._reliability_metric.requires_golden():
            self._populate_golden()

        model = self._inject_faults()
        bitmask = self._compare_bitwise(model)
        result = self._infer(model)
        self._record_result(result, bitmask)


def _batch_critical_sdc(
    logits: Tensor, golden_classifications: Tensor
) -> BatchReliability:
    """Compute the critical SDC of a result. Used for ReliabilityMetric.CriticalSdc."""

    classifications = logits.argmax(dim=1)

    assert golden_classifications.shape == classifications.shape
    # bool is a subclass of int, so sum converts bools to ints.
    correct = int((classifications == golden_classifications).sum().item())
    total = golden_classifications.numel()

    return BatchReliability(correct=correct, total=total)


def _batch_sdc(logits: Tensor, golden_logits: Tensor) -> BatchReliability:
    """Compute the SDC of a result. Used for ReliabilityMetric.Sdc."""

    assert golden_logits.shape == logits.shape
    # bool is a subclass of int, so sum converts bools to ints.
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
