"""Progress reporting for long-running operations.

`Progress` reports on named stages of work via the standard-library `logging`
module, throttled to avoid spamming. Pass `None` anywhere a `Progress` is
accepted to disable reporting with no overhead. Use the `stage` function
(rather than calling `Progress.stage` directly) at any call site that accepts
an optional `Progress` - it handles the `None` case and returns a
`StageHandle` context manager whose `advance` reports incremental progress.
"""

from faultforge._internal.progress import Progress, StageHandle, stage

__all__ = [
    "Progress",
    "StageHandle",
    "stage",
]
