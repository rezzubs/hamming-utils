"""Tests for faultforge._internal.io."""

from pathlib import Path

from faultforge._internal.io import is_compressed, open_text

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def test_is_compressed_true_for_zstd_magic_prefix(tmp_path: Path):
    path = tmp_path / "data"
    path.write_bytes(_ZSTD_MAGIC + b"whatever follows")
    assert is_compressed(path) is True


def test_is_compressed_false_for_plain_text(tmp_path: Path):
    path = tmp_path / "data"
    path.write_text("just some plain text")
    assert is_compressed(path) is False


def test_is_compressed_false_for_file_shorter_than_magic(tmp_path: Path):
    path = tmp_path / "data"
    path.write_bytes(b"\x28\xb5")
    assert is_compressed(path) is False


def test_open_text_round_trip_uncompressed(tmp_path: Path):
    path = tmp_path / "data"
    with open_text(path, "wt", compressed=False) as f:
        f.write("hello world")

    assert path.read_text() == "hello world"
    with open_text(path, "rt", compressed=False) as f:
        assert f.read() == "hello world"


def test_open_text_round_trip_compressed(tmp_path: Path):
    path = tmp_path / "data"
    with open_text(path, "wt", compressed=True) as f:
        f.write("hello world")

    assert path.read_bytes()[:4] == _ZSTD_MAGIC
    with open_text(path, "rt", compressed=True) as f:
        assert f.read() == "hello world"
