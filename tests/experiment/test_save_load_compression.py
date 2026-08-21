"""Tests for Experiment.save/load compressed (zstd) round-trip behavior."""

import os
from os import PathLike
from pathlib import Path

import pytest
from faultforge.experiment import AdditionalRuns, SaveConfig

from .conftest import make

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def test_save_compressed_writes_zstd_magic_bytes(tmp_path: Path):
    exp = make([1.0, 2.0])
    path = tmp_path / "experiment.json"
    exp.save(path, compressed=True)
    assert path.read_bytes()[:4] == _ZSTD_MAGIC


def test_save_compressed_content_round_trips(tmp_path: Path):
    exp = make([1.0, 2.0, 3.0])
    path = tmp_path / "experiment.json"
    exp.save(path, compressed=True)
    loaded = make()
    loaded.load_from(path)
    assert loaded.run_count() == exp.run_count()


def test_save_atomic_compressed_writes_correct_content(tmp_path: Path):
    exp = make([10.0, 20.0])
    path = tmp_path / "experiment.json"
    exp.save_atomic(path, compressed=True)
    assert path.read_bytes()[:4] == _ZSTD_MAGIC
    loaded = make()
    loaded.load_from(path)
    assert loaded.run_count() == exp.run_count()


def test_save_atomic_compressed_uses_tempfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exp = make([1.0])
    path = tmp_path / "out.json"
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def recording_replace(src: str | PathLike[str], dst: str | PathLike[str]) -> None:
        replaced.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", recording_replace)
    exp.save_atomic(path, compressed=True)

    assert len(replaced) == 1
    src, dst = replaced[0]
    assert src != dst
    assert dst == str(path.expanduser())


def test_load_from_auto_detects_compressed_regardless_of_name(tmp_path: Path):
    # Deliberately named ".json", not ".zst" - detection must be by content.
    exp = make([1.0, 2.0])
    path = tmp_path / "out.json"
    exp.save_atomic(path, compressed=True)

    loaded = make()
    loaded.load_from(path)
    assert loaded.run_count() == exp.run_count()


def test_load_from_auto_detects_uncompressed(tmp_path: Path):
    exp = make([1.0, 2.0])
    path = tmp_path / "out.zst"
    exp.save(path, compressed=False)

    loaded = make()
    loaded.load_from(path)
    assert loaded.run_count() == exp.run_count()


def test_save_warns_on_compressed_without_zst_extension(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    exp = make([1.0])
    path = tmp_path / "out.json"
    with caplog.at_level("WARNING"):
        exp.save(path, compressed=True)
    assert any("out.json" in record.message for record in caplog.records)
    assert path.exists()


def test_save_warns_on_zst_extension_without_compressed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    exp = make([1.0])
    path = tmp_path / "out.zst"
    with caplog.at_level("WARNING"):
        exp.save(path, compressed=False)
    assert any("out.zst" in record.message for record in caplog.records)
    assert path.read_bytes()[:4] != _ZSTD_MAGIC


def test_save_no_warning_when_extension_matches_compressed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    exp = make([1.0])
    with caplog.at_level("WARNING"):
        exp.save(tmp_path / "out.zst", compressed=True)
        exp.save(tmp_path / "out.json", compressed=False)
    assert caplog.records == []


def test_save_config_compressed_defaults_to_false():
    assert SaveConfig(path="x", interval_seconds=None).compressed is False


def test_run_loop_saves_compressed_when_configured(tmp_path: Path):
    exp = make()
    path = tmp_path / "experiment.json"
    exp.run_loop(
        stop_conditions=[AdditionalRuns(3)],
        save_config=SaveConfig(path=path, interval_seconds=None, compressed=True),
    )
    assert path.read_bytes()[:4] == _ZSTD_MAGIC
    loaded = make()
    loaded.load_from(path)
    assert loaded.run_count() == 3
