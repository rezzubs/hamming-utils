"""Shared test doubles for the encoded_memory experiment tests."""

import json
from typing import override

import torch
from encoded_memory import (
    EncodedFaultInjection,
    ReliabilityMetric,
)
from faultforge import Fingerprint
from faultforge.dataset import BatchedDataset, DeviceLike
from faultforge.encoding import IdentityEncoder
from faultforge.loading import ModelBundle
from faultforge.progress import Progress
from torch import nn
from torch.utils.data import TensorDataset

# `_` prefixed to not interpret it as a Test class.


class _FakeBundle(ModelBundle):
    """A tiny in-memory model/dataset bundle, just enough to drive `EncodedFaultInjection`."""

    def __init__(
        self, in_features: int, out_features: int, batch_size: int, num_batches: int
    ) -> None:
        self._in_features = in_features
        self._out_features = out_features
        self._batch_size = batch_size
        self._num_batches = num_batches

    @override
    def load_model(
        self,
        device: DeviceLike,
        *,
        dtype: torch.dtype = torch.float32,
        progress: Progress | None = None,
    ) -> nn.Module:
        return nn.Linear(self._in_features, self._out_features).to(
            device=device, dtype=dtype
        )

    @override
    def load_dataset(
        self, batch_size: int, device: DeviceLike, *, progress: Progress | None = None
    ) -> BatchedDataset:
        n = self._batch_size * self._num_batches
        inputs = torch.randn(n, self._in_features)
        targets = torch.randint(0, self._out_features, (n,))
        dataset = TensorDataset(inputs, targets)
        return BatchedDataset.from_dataset(dataset, batch_size, device)

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint(kind="fake_bundle")


def _make_experiment(
    *,
    compare_bitwise: bool,
    faults: int | float = 1,
    golden_is_encoded: bool = False,
    dataset_batch_limit: int | None = None,
    reliability_metric: ReliabilityMetric = ReliabilityMetric.Accuracy,
    dtype: torch.dtype = torch.float32,
    fault_summary: bool = False,
) -> EncodedFaultInjection:
    bundle = _FakeBundle(in_features=4, out_features=3, batch_size=2, num_batches=2)
    return EncodedFaultInjection(
        bundle,
        IdentityEncoder(),
        reliability_metric,
        golden_is_encoded=golden_is_encoded,
        faults=faults,
        compare_bitwise=compare_bitwise,
        fault_summary=fault_summary,
        dataset_batch_limit=dataset_batch_limit,
        batch_size=2,
        dtype=dtype,
    )


def _result(experiment: EncodedFaultInjection) -> dict:
    return json.loads(experiment.serialize())["result"]
