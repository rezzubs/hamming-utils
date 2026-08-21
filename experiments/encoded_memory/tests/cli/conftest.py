"""Shared fixtures for the `encoded_memory` CLI tests.

`_FakeBundle` is a tiny in-memory model/dataset bundle, just enough to drive a
real `EncodedFaultInjection` so tests exercise the genuine save/load path rather
than hand-crafted fingerprints. The `save_result`/`make_configuration` factory
fixtures wrap it so tests can lay out real result files on disk cheaply.
"""

from pathlib import Path
from typing import override

import pytest
import torch
from encoded_memory import (
    EncodedFaultInjection,
    ReliabilityMetric,
)
from faultforge import Fingerprint
from faultforge.dataset import BatchedDataset, DeviceLike
from faultforge.encoding import IdentityEncoder
from faultforge.loading import ModelBundle
from encoded_memory.results import (
    Configuration,
    build_configurations,
    load_results,
)
from torch import nn
from torch.utils.data import TensorDataset


class _FakeBundle(ModelBundle):
    """A tiny in-memory model/dataset bundle, just enough to drive `EncodedFaultInjection`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        batch_size: int,
        num_batches: int,
        *,
        model: str = "toynet",
        dataset: str | None = None,
    ) -> None:
        self._in_features = in_features
        self._out_features = out_features
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._model = model
        self._dataset = dataset

    @override
    def load_model(
        self,
        device: DeviceLike,
        *,
        dtype: torch.dtype = torch.float32,
        progress=None,
    ) -> nn.Module:
        return nn.Linear(self._in_features, self._out_features).to(
            device=device, dtype=dtype
        )

    @override
    def load_dataset(
        self, batch_size: int, device: DeviceLike, *, progress=None
    ) -> BatchedDataset:
        n = self._batch_size * self._num_batches
        inputs = torch.randn(n, self._in_features)
        targets = torch.randint(0, self._out_features, (n,))
        dataset = TensorDataset(inputs, targets)
        return BatchedDataset.from_dataset(dataset, batch_size, device)

    @override
    def fingerprint(self) -> Fingerprint:
        scalars = {"model": self._model}
        if self._dataset is not None:
            scalars["dataset"] = self._dataset
        return Fingerprint(kind="fake_bundle", scalars=scalars)


@pytest.fixture
def save_result():
    """Factory: run a real `EncodedFaultInjection` and save it to `path`."""

    def _save(
        path: Path,
        *,
        faults: float | int = 0.05,
        runs: int = 1,
        model: str = "toynet",
        dataset: str | None = None,
        dtype: torch.dtype = torch.float32,
        metric: ReliabilityMetric = ReliabilityMetric.Accuracy,
        compare_bitwise: bool = True,
    ) -> Path:
        bundle = _FakeBundle(8, 4, 4, 4, model=model, dataset=dataset)
        experiment = EncodedFaultInjection(
            bundle,
            IdentityEncoder(),
            metric,
            faults=faults,
            compare_bitwise=compare_bitwise,
            batch_size=4,
            dtype=dtype,
        )
        for _ in range(runs):
            experiment.run()
        path.parent.mkdir(parents=True, exist_ok=True)
        experiment.save(path)
        return path

    return _save


@pytest.fixture
def make_configuration(save_result):
    """Factory: save several bit error rates under `directory` and return the
    single `Configuration` they cluster into."""

    def _make(
        directory: Path,
        *,
        label: str,
        rates: tuple[float, ...] = (0.01, 0.05, 0.1),
        runs: int = 2,
        **kwargs,
    ) -> Configuration:
        for rate in rates:
            save_result(
                directory / f"ber_{rate}.json", faults=rate, runs=runs, **kwargs
            )
        loaded = load_results([directory])
        overrides = {path: label for path, _ in loaded}
        configurations = build_configurations(loaded, label_overrides=overrides)
        assert len(configurations) == 1
        return configurations[0]

    return _make
