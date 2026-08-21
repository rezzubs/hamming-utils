"""File I/O helpers."""

from compression import zstd
from os import PathLike
from pathlib import Path
from typing import IO, Literal

type AnyPath = str | PathLike[str]

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
"""The first 4 bytes of any zstd-compressed frame."""


def is_compressed(path: AnyPath) -> bool:
    """Whether the file at `path` is zstd-compressed.

    Detected by sniffing the first 4 bytes for the zstd frame magic number
    rather than trusting the file name, so callers work regardless of
    whatever extension (or lack of one) the file happens to have.
    """
    with open(Path(path).expanduser(), "rb") as f:
        return f.read(len(_ZSTD_MAGIC)) == _ZSTD_MAGIC


def open_text(path: AnyPath, mode: Literal["rt", "wt"], *, compressed: bool) -> IO[str]:
    """Open `path` in text mode, transparently through zstd if `compressed`.

    A thin wrapper around builtin `open`/`compression.zstd.open` so callers
    don't have to duplicate the compressed/uncompressed branch themselves.
    """
    resolved = Path(path).expanduser()
    if compressed:
        return zstd.open(resolved, mode)
    return open(resolved, mode)
