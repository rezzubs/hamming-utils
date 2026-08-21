"""Tests for `encoded_memory.plots.group_key`."""

import pytest
from faultforge import Fingerprint
from encoded_memory.plots import GroupBy, group_key


def test_group_key_ungrouped_is_none():
    fingerprint = Fingerprint(kind="k", scalars={"dtype": "f32"})
    assert group_key(GroupBy.Ungrouped, fingerprint) is None


def test_group_key_dtype_and_metric():
    fingerprint = Fingerprint(
        kind="k", scalars={"dtype": "f16", "reliability_metric": "sdc"}
    )
    assert group_key(GroupBy.Dtype, fingerprint) == "f16"
    assert group_key(GroupBy.Metric, fingerprint) == "sdc"


def test_group_key_model_and_dataset():
    fingerprint = Fingerprint(
        kind="k",
        children={
            "bundle": [
                Fingerprint(
                    kind="cifar", scalars={"model": "resnet20", "dataset": "cifar10"}
                )
            ]
        },
    )
    assert group_key(GroupBy.Model, fingerprint) == "resnet20"
    assert group_key(GroupBy.Dataset, fingerprint) == "cifar10"


def test_group_key_dataset_raises_when_absent():
    fingerprint = Fingerprint(
        kind="k",
        children={"bundle": [Fingerprint(kind="imagenet", scalars={"model": "vit"})]},
    )
    with pytest.raises(ValueError, match="dataset"):
        group_key(GroupBy.Dataset, fingerprint)
