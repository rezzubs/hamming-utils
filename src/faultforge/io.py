"""File I/O helpers.

`open_text`/`is_compressed` are the same helpers `Experiment.save`/
`load_from` use internally to read/write save files transparently through
zstd compression - useful directly when writing your own save-file format.
"""

from faultforge._internal.io import AnyPath, is_compressed, open_text

__all__ = [
    "AnyPath",
    "is_compressed",
    "open_text",
]
