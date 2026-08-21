"""Validate that BackendModel(model, TorchBackend()) reproduces baseline accuracy.

Proof of Phase 0's "Done when" criterion from systolic-roadmap.md. Not part
of pytest/CI: downloads a real model and dataset. Run by hand.
"""

import copy

import torch
from faultforge.dataset import DEFAULT_BATCH_SIZE, DEFAULT_DEVICE, BatchedDataset
from faultforge.loading import Cifar, CifarDataset, CifarModel
from systolic import BackendModel, TorchBackend
from torch import nn


def _accuracy(model: nn.Module, dataset: BatchedDataset) -> float:
    dataset.reset()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataset:
            logits = model.forward(batch.inputs)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == batch.targets).sum().item())
            total += batch.targets.shape[0]
    return correct / total


def main() -> None:
    # ResNet20: groups=1 throughout. MappedConv2d does not support grouped
    # convolutions yet (see systolic-roadmap.md's open questions), so
    # MobileNetV2/ShuffleNetV2 variants of this bundle can't be used here.
    bundle = Cifar(CifarModel.ResNet20, CifarDataset.Cifar10)
    reference = bundle.load_model(DEFAULT_DEVICE).eval()
    dataset = bundle.load_dataset(DEFAULT_BATCH_SIZE, DEFAULT_DEVICE)

    reference_accuracy = _accuracy(reference, dataset)

    wrapped = BackendModel(copy.deepcopy(reference), TorchBackend()).eval()
    wrapped_accuracy = _accuracy(wrapped, dataset)

    print(f"reference accuracy: {reference_accuracy:.4f}")
    print(f"wrapped accuracy:   {wrapped_accuracy:.4f}")
    assert reference_accuracy == wrapped_accuracy, (
        "BackendModel(model, TorchBackend()) accuracy diverged from the reference model"
    )


if __name__ == "__main__":
    main()
