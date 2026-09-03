"""Progress reporting for long-running operations.

`Progress` reports on named stages of work via the standard-library `logging`
module, throttled to avoid spamming. Pass `None` anywhere a `Progress` is
accepted to disable reporting with no overhead. Use the `stage` function
(rather than calling `Progress.stage` directly) at any call site that accepts
an optional `Progress` - it handles the `None` case and returns a
`StageHandle` context manager whose `advance` reports incremental progress.

Stages nest: opening one while another is already open makes it a child, and
log lines render the whole open chain.

`advance` only ever moves the counter of the stage it's called on, so pass a
`total` only when the caller opening the stage also drives the work producing
those units. A stage that merely spans work driven by someone else's control
flow should be opened without one, and reports elapsed time alone.
"""

from faultforge._internal.progress import Progress, StageHandle, stage

__all__ = [
    "Progress",
    "StageHandle",
    "stage",
]
