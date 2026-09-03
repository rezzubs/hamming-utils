"""Tests for BackendModel: full-forward correctness and the in-place-mutation contract."""

import copy
import logging

import pytest
import torch
from faultforge.progress import Progress
from torch import nn

from systolic import BackendModel, MappedConv2d, MappedLinear, TorchBackend


def _make_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 5),
    )


def test_backend_model_forward_matches_plain() -> None:
    model = _make_model()
    reference = copy.deepcopy(model)
    wrapped = BackendModel(copy.deepcopy(model), TorchBackend())

    x = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        expected = reference.forward(x)
        actual = wrapped.forward(x)

    torch.testing.assert_close(actual, expected)


def test_backend_model_bare_linear_at_top_level() -> None:
    model = nn.Linear(4, 3)
    reference = copy.deepcopy(model)
    wrapped = BackendModel(copy.deepcopy(model), TorchBackend())

    x = torch.randn(2, 4)

    with torch.no_grad():
        expected = reference.forward(x)
        actual = wrapped.forward(x)

    torch.testing.assert_close(actual, expected)


def test_backend_model_mutates_wrapped_model_in_place() -> None:
    model = _make_model()
    original_linear = model[3]
    original_conv = model[0]

    BackendModel(model, TorchBackend())

    mapped_linear = model[3]
    mapped_conv = model[0]
    assert isinstance(mapped_linear, MappedLinear)
    assert isinstance(mapped_conv, MappedConv2d)
    assert mapped_linear.inner is original_linear
    assert mapped_conv.inner is original_conv


def test_backend_model_forward_logs_layers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    model = _make_model()
    wrapped = BackendModel(
        copy.deepcopy(model), TorchBackend(), progress=Progress(min_log_interval=0.0)
    )
    x = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        wrapped.forward(x)

    messages = [r.message for r in caplog.records]
    # The Sequential's index names ("0" for Conv2d, "3" for Linear), not the
    # internal "_wrapped" attribute BackendModel stores the model under, each
    # followed by its position among the model's two mapped layers.
    assert not any("_wrapped" in m for m in messages)
    assert any("0 [1/2] Conv2d" in m for m in messages)
    assert any("3 [2/2] Linear" in m for m in messages)


def test_backend_model_forward_no_progress_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    model = _make_model()
    wrapped = BackendModel(copy.deepcopy(model), TorchBackend())
    x = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        wrapped.forward(x)

    assert caplog.records == []
