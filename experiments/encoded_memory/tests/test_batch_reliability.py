"""Tests for per-metric batch reliability functions."""

import torch
from encoded_memory.experiment import (
    BatchReliability,
    _batch_accuracy,
    _batch_accuracy_degradation,
    _batch_critical_sdc,
    _batch_sdc,
)


def test_batch_accuracy_counts_matching_predictions():
    logits = torch.tensor(
        [
            [0.1, 0.9],  # predicts class 1
            [0.8, 0.2],  # predicts class 0
            [0.3, 0.7],  # predicts class 1
        ]
    )
    targets = torch.tensor([1, 0, 0])  # last one is wrong

    assert _batch_accuracy(logits, targets) == BatchReliability(correct=2, total=3)


def test_batch_accuracy_degradation_is_golden_minus_faulty_correct():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])  # predicts 1, 0, 1
    golden_classifications = torch.tensor([1, 0, 0])
    targets = torch.tensor([1, 0, 0])

    # faulty correct: [1, 0, 1] vs [1, 0, 0] -> 2
    # golden correct: [1, 0, 0] vs [1, 0, 0] -> 3
    # degradation: golden_correct - correct = 1
    result = _batch_accuracy_degradation(logits, golden_classifications, targets)
    assert result == BatchReliability(correct=1, total=3)


def test_batch_sdc_counts_matching_logits_elementwise():
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    golden_logits = torch.tensor([[1.0, 0.0], [3.0, 0.0]])

    # elementwise equality: [[T, F], [T, F]] -> 2 correct out of 4
    result = _batch_sdc(logits, golden_logits)
    assert result == BatchReliability(correct=2, total=4)


def test_batch_critical_sdc_counts_matching_top1_predictions():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])  # predicts 1, 0, 1
    golden_classifications = torch.tensor([1, 0, 0])  # matches the first two only

    result = _batch_critical_sdc(logits, golden_classifications)
    assert result == BatchReliability(correct=2, total=3)
