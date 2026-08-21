"""Tests for EncodedFaultInjection.discard_bitmasks and discard_bitmasks_in_file."""

import json
from compression import zstd

from encoded_memory import discard_bitmasks_in_file

from .conftest import _make_experiment, _result

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def test_discard_bitmasks_converts_to_simple():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()
    before = _result(experiment)

    experiment.discard_bitmasks()
    after = _result(experiment)

    assert after["kind"] == "simple"
    assert after["results"] == [run["correct_count"] for run in before["results"]]


def test_discard_bitmasks_is_noop_for_simple():
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()

    experiment.discard_bitmasks()

    assert _result(experiment)["kind"] == "simple"


def test_discard_bitmasks_updates_fingerprint_compare_bitwise_scalar():
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    experiment.discard_bitmasks()

    scalars = json.loads(experiment.serialize())["fingerprint"]["scalars"]
    assert scalars["compare_bitwise"] is False


def test_deserialize_after_discard_bitmasks_matches_simple_experiment():
    detailed = _make_experiment(compare_bitwise=True)
    detailed.run()
    detailed.discard_bitmasks()
    serialized = detailed.serialize()

    # Must not raise FingerprintError: the discarded result's fingerprint
    # should now agree with an experiment that was never recording bitmasks.
    simple = _make_experiment(compare_bitwise=False)
    simple.deserialize(serialized)
    assert json.loads(simple.serialize())["result"]["kind"] == "simple"


def test_discard_bitmasks_in_file(tmp_path):
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()
    before = _result(experiment)

    path = tmp_path / "result.json"
    experiment.save(path)

    discard_bitmasks_in_file(path)

    saved = json.loads(path.read_text())
    assert saved["result"]["kind"] == "simple"
    assert saved["result"]["results"] == [
        run["correct_count"] for run in before["results"]
    ]
    assert saved["fingerprint"]["scalars"]["compare_bitwise"] is False


def test_discard_bitmasks_in_file_is_noop_for_simple_results(tmp_path):
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save(path)
    original = path.read_text()

    discard_bitmasks_in_file(path)

    assert path.read_text() == original


def test_load_from_after_discard_bitmasks_in_file(tmp_path):
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save(path)
    discard_bitmasks_in_file(path)

    # Must not raise FingerprintError: the discarded file's fingerprint should
    # now agree with an experiment that was never recording bitmasks.
    simple = _make_experiment(compare_bitwise=False)
    simple.load_from(path)
    assert json.loads(simple.serialize())["result"]["kind"] == "simple"


def test_discard_bitmasks_in_file_preserves_compressed_format(tmp_path):
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()
    before = _result(experiment)

    path = tmp_path / "result.json"
    experiment.save_atomic(path, compressed=True)

    discard_bitmasks_in_file(path)

    assert path.read_bytes()[:4] == _ZSTD_MAGIC
    with zstd.open(path, "rt") as f:
        saved = json.loads(f.read())
    assert saved["result"]["kind"] == "simple"
    assert saved["result"]["results"] == [
        run["correct_count"] for run in before["results"]
    ]
    assert saved["fingerprint"]["scalars"]["compare_bitwise"] is False


def test_discard_bitmasks_in_file_preserves_uncompressed_format(tmp_path):
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save(path)

    discard_bitmasks_in_file(path)

    assert path.read_bytes()[:4] != _ZSTD_MAGIC


def test_discard_bitmasks_in_file_is_noop_for_simple_results_compressed(tmp_path):
    experiment = _make_experiment(compare_bitwise=False)
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save_atomic(path, compressed=True)
    original = path.read_bytes()

    discard_bitmasks_in_file(path)

    assert path.read_bytes() == original


def test_load_from_after_discard_bitmasks_in_file_compressed(tmp_path):
    experiment = _make_experiment(compare_bitwise=True)
    experiment.run()

    path = tmp_path / "result.json"
    experiment.save_atomic(path, compressed=True)
    discard_bitmasks_in_file(path)

    # Must not raise FingerprintError: the discarded file's fingerprint should
    # now agree with an experiment that was never recording bitmasks.
    simple = _make_experiment(compare_bitwise=False)
    simple.load_from(path)
    assert json.loads(simple.serialize())["result"]["kind"] == "simple"
