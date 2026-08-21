"""Tests for EncodedFaultInjection.serialize/deserialize round-trip behavior."""

from .conftest import _make_experiment


def test_serialize_deserialize_round_trip_detailed():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()
    serialized = experiment.serialize()

    reloaded = _make_experiment(compare_bitwise=True)
    reloaded.deserialize(serialized)

    assert reloaded.serialize() == serialized


def test_serialize_deserialize_round_trip_simple():
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()
    serialized = experiment.serialize()

    reloaded = _make_experiment(compare_bitwise=False)
    reloaded.deserialize(serialized)

    assert reloaded.serialize() == serialized
