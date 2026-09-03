"""Tests for `MappingCache`."""

import torch

from systolic.backend import MappingCache


def test_mapping_cache_reuses_mapping_for_same_shape() -> None:
    cache = MappingCache(nrows=4, ncols=4)
    weights_a = torch.randn(3, 4)
    weights_b = torch.randn(3, 4)

    mapping_a = cache.get(weights_a)
    mapping_b = cache.get(weights_b)

    assert mapping_a is mapping_b


def test_mapping_cache_builds_separately_per_shape() -> None:
    cache = MappingCache(nrows=4, ncols=4)
    small = cache.get(torch.randn(2, 4))
    large = cache.get(torch.randn(3, 4))

    assert small is not large
