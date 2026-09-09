import abc
from dataclasses import dataclass
from typing import final, override

from torch import Tensor


class Metric[R](abc.ABC):
    """A metric for evaluating the output of a DNN.

    In the end all metrics need to produce a numeric [`score`] but can use any
    intermediate representation (the generic type `R`) which is later turned
    into a scalar float by [`score`].

    Metric data (the result type `R`) is created by [`evaluate_batch`] which
    is expected to be called on every iteration over the [batches] of the
    [dataset]. [`evaluate_batch`] has three parameters:
    - `model_output` - The output of a fault-injected [`Module`] for a single
      batch of data.
    - `golden` - The output of a non-faulty [`Module`] for the same batch
      of data, passed through [`preprocess_golden`]. See the [Golden
      outputs](#golden-outputs) section below.
    - `targets` - the `targets` field from the current [`DataBatch`]. The
      expected output of the model as defined by the [dataset].

    [`evaluate_batch`] processes a result for just one batch. The final metric
    is produced by repeated runs of [`accumulate`] which takes two instances of
    the result (`R`) as arguments. The caller is expected to feed the combined
    result as the `existing` argument of [`accumulate`] and the new result as
    the [`new`] argument.

    # Golden outputs

    Computing `golden` may not be required for all metrics so the
    [`requies_golden`] method can be used to signal whether golden outputs
    should be computed. Metrics which don't need golden outputs can ignore the
    `golden` parameter on [`evaluate_batch`].

    In some cases you may want to run a processing step on the golden outputs
    which is the same for every run. In this case the metric can override
    [`preprocess_golden`] to define that processing step. This enables the
    caller to cache the processed golden values so they don't have to be
    recomputed every time [`evaluate_batch`] is called. The caller is always
    expected to pass the golden outputs through [`preprocess_golden`], the
    default implementation just returns the values without altering them.


    [`accumulate`]: Metric.accumulate
    [`evaluate_batch`]: Metric.evaluate_batch
    [`preprocess_golden`]: Metric.preprocess_golden
    [`requires_golden`]: Metric.requires_golden
    [`score`]: Metric.score

    [`DataBatch`]: faultforge.dataset.DataBatch
    [`Module`]: torch.nn.Module
    [batches]: faultforge.dataset.DataBatch
    [dataset]: faultforge.dataset.BatchedDataset
    """

    @abc.abstractmethod
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> R:
        """Create result data for this metric from model output for one dataset batch.

        See the docstring on [`Metric`] for details on the parameters.
        """

    def preprocess_golden(self, golden: Tensor) -> Tensor:
        """Process the golden outputs of the model before they are passed to [`evaluate_batch`].

        See [Metric#golden-outputs] for details.

        [`evaluate_batch`]: Metric.evaluate_batch
        """
        return golden

    @abc.abstractmethod
    def requires_golden(self) -> bool:
        """Whether `evaluate_batch` requires golden inputs.

        The return value of this function assumed to be constant for any
        specific `Metric`.

        This is a signal for the user of the metric of whether to compute golden
        outputs. If this function returns false then the caller may pass an
        empty tensor to [`evaluate_batch`] because it's implied that it will be
        ignored anyway.

        [`evaluate_batch`]: Metric.evaluate_batch
        """

    @abc.abstractmethod
    def accumulate(self, existing: R, new: R) -> R:
        """Combine the result of the current iteration with existing ones.

        - `existing` is the combined result of the already completed iterations
        - `new` is the result of the current iterations
        """

    @abc.abstractmethod
    def score(self, result: R) -> float:
        """Give a numeric "score" for the result."""


@final
@dataclass(frozen=True, slots=True)
class AccuracyResult:
    """A result type for metrics which count correct runs over the total number of iterations."""

    correct_count: int
    """Number of times the top-1 prediction matched `targets`."""
    total_count: int
    """Total number of predictions made."""


@final
class Accuracy(Metric[AccuracyResult]):
    """Percentage of top-1 predictions vs dataset targets over all iterations.

    Top-1 in this case means that we take the `argmax` over a  model's
    output logits.

    Example usees:
    - image classification models
    - encoder based transformers.
    """

    @override
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> AccuracyResult:
        _ = batch_golden

        top1_logit = top1_logit_from_output(batch_model_output)

        # NOTE: sum implicitly converts bools to ints.
        correct_count = (top1_logit == batch_targets).sum().item()
        assert isinstance(correct_count, int)

        return AccuracyResult(
            correct_count=correct_count,
            total_count=top1_logit.numel(),
        )

    @override
    def requires_golden(self) -> bool:
        return False

    @override
    def accumulate(
        self, existing: AccuracyResult, new: AccuracyResult
    ) -> AccuracyResult:
        return AccuracyResult(
            correct_count=existing.correct_count + new.correct_count,
            total_count=existing.total_count + new.total_count,
        )

    @override
    def score(self, result: AccuracyResult) -> float:
        return ratio_to_percent(result.correct_count / result.total_count)


@final
@dataclass(frozen=True, slots=True)
class AccuracyDegradationResult:
    """The difference of accuracy between faulty and golden runs.

    Positive if the accuracy dropped due to fault injection.
    """

    correct_count_faulty: int
    """Number of times the top-1 prediction matched `targets` for the faulty run."""
    correct_count_golden: int
    """Number of times the top-1 prediction matched `targets` for the non-faulty (golden) run."""
    total_count: int
    """Total number of predictions made."""


class AccuracyDegradation(Metric[AccuracyDegradationResult]):
    """The difference of [`Accuracy`] between faulty and golden runs.

    Positive if the accuracy dropped due to fault injection.

    Example usees:
    - image classification models
    - encoder based transformers.
    """

    @override
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> AccuracyDegradationResult:
        _ = batch_golden

        top1_logit = top1_logit_from_output(batch_model_output)

        # NOTE: sum implicitly converts bools to ints.
        correct_count_faulty = (top1_logit == batch_targets).sum().item()
        assert isinstance(correct_count_faulty, int)

        # NOTE: golden is preprocessed
        correct_count_golden = (batch_golden == batch_targets).sum().item()
        assert isinstance(correct_count_golden, int)

        return AccuracyDegradationResult(
            correct_count_faulty=correct_count_faulty,
            correct_count_golden=correct_count_golden,
            total_count=top1_logit.numel(),
        )

    @override
    def preprocess_golden(self, golden: Tensor) -> Tensor:
        return top1_logit_from_output(golden)

    @override
    def requires_golden(self) -> bool:
        return True

    @override
    def accumulate(
        self, existing: AccuracyDegradationResult, new: AccuracyDegradationResult
    ) -> AccuracyDegradationResult:
        return AccuracyDegradationResult(
            correct_count_faulty=existing.correct_count_faulty
            + new.correct_count_faulty,
            correct_count_golden=existing.correct_count_golden
            + new.correct_count_golden,
            total_count=existing.total_count + new.total_count,
        )

    @override
    def score(self, result: AccuracyDegradationResult) -> float:
        return ratio_to_percent(
            (result.correct_count_golden - result.correct_count_faulty)
            / result.total_count
        )


@final
@dataclass(slots=True, frozen=True)
class SdcResult:
    non_matching_count: int
    """Number of matched elements"""
    total_count: int
    """Number of total elements"""


class Sdc(Metric[SdcResult]):
    """Silent Data Corruption.

    Percentage of output values that have changed in the case of fault injection
    vs the golden outputs. Does not care about what is actually predicted
    (unlike Top1Sdc). Works with any model output shape.

    Example usees:
    - basically any model output.
    """

    @override
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> SdcResult:
        _ = batch_targets

        if batch_model_output.shape != batch_golden.shape:
            raise ValueError(
                f"Shape mismatch between shape of model_output ({batch_model_output.shape}) and golden ({batch_golden.shape})"
            )

        # NOTE: sum implicitly converts bools to ints.
        non_matching_count = (batch_model_output == batch_golden).sum().item()
        assert isinstance(non_matching_count, int)

        return SdcResult(
            non_matching_count=non_matching_count,
            total_count=batch_model_output.numel(),
        )

    @override
    def requires_golden(self) -> bool:
        return True

    @override
    def accumulate(self, existing: SdcResult, new: SdcResult) -> SdcResult:
        return SdcResult(
            non_matching_count=existing.non_matching_count + new.non_matching_count,
            total_count=existing.total_count + new.total_count,
        )

    @override
    def score(self, result: SdcResult) -> float:
        return ratio_to_percent(result.non_matching_count / result.total_count)


class Top1Sdc(Metric[SdcResult]):
    """Silent Data Corruption for the prediction (top-1 logit).

    Percentage of output values that have changed in the case of fault injection
    vs the golden outputs. Only compares the actual prediction (`argmax` over
    the logits). See [`Sdc`] for the general version.

    Example usees:
    - image classification models
    - encoder based transformers.
    """

    @override
    def evaluate_batch(
        self,
        batch_model_output: Tensor,
        batch_golden: Tensor,
        batch_targets: Tensor,
    ) -> SdcResult:
        top1_logits = top1_logit_from_output(batch_model_output)

        return Sdc().evaluate_batch(top1_logits, batch_golden, batch_targets)

    @override
    def preprocess_golden(self, golden: Tensor) -> Tensor:
        return top1_logit_from_output(golden)

    @override
    def requires_golden(self) -> bool:
        return Sdc().requires_golden()

    @override
    def accumulate(self, existing: SdcResult, new: SdcResult) -> SdcResult:
        return Sdc().accumulate(existing, new)

    @override
    def score(self, result: SdcResult) -> float:
        return Sdc().score(result)


def ratio_to_percent(ratio: float) -> float:
    return ratio * 100


def top1_logit_from_output(batch_model_output: Tensor) -> Tensor:
    """Extract the top-1 logit from the model output (batched).

    Input n-d, output (n-1)-d
    """
    if len(batch_model_output.shape) < 2:
        raise ValueError(
            f"Logit extraction expects >=2 dimensional tensors, got shape {batch_model_output.shape}"
        )

    return batch_model_output.argmax(dim=1)
