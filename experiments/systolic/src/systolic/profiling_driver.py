"""Walk a model's mapped layers over a dataset subsample, producing a profiling artifact."""

import torch
from faultforge.dataset import DEFAULT_DEVICE, DeviceLike
from faultforge.loading import DEFAULT_DTYPE, ModelBundle
from faultforge.progress import Progress, stage

from systolic._rust import ProfilingArtifact
from systolic.model import BackendModel
from systolic.profiling import ProfilingMetadata
from systolic.profiling_backend import ProfilingBackend


def run_profiling(
    bundle: ModelBundle,
    array: tuple[int, int],
    capacity: int,
    *,
    subsample_size: int,
    seed: int = 0,
    device: DeviceLike = DEFAULT_DEVICE,
    dtype: torch.dtype = DEFAULT_DTYPE,
    progress: Progress | None = None,
) -> tuple[ProfilingArtifact, ProfilingMetadata]:
    """Profile a model's per-PE logic inputs over a random dataset subsample.

    Walks every mapped `nn.Linear`/`nn.Conv2d` layer in one forward pass over
    `subsample_size` images. The subsample is sampled uniformly at random
    (via `bundle.load_dataset(shuffle=True)`), not stratified, so the profile
    reflects the real class mixture a deployed model sees. `seed` reproduces
    both that sampling and the `Profiler`'s own reservoir sampling.
    """
    if dtype != torch.float32:
        raise ValueError(f"run_profiling only supports float32, got {dtype}")

    nrows, ncols = array
    model = bundle.load_model(device, dtype=dtype, progress=progress)
    backend = ProfilingBackend(nrows, ncols, capacity, seed)
    profiling_model = BackendModel(model, backend)

    # One batch of exactly subsample_size.
    dataset = bundle.load_dataset(
        subsample_size, device, shuffle=True, seed=seed, progress=progress
    )

    with stage(progress, "Profiling"), torch.no_grad():
        batch = next(dataset)
        profiling_model.forward(batch.inputs.to(dtype=dtype))

    metadata = ProfilingMetadata(
        array_rows=nrows, array_cols=ncols, capacity=capacity, seed=seed
    )
    return backend.artifact(), metadata
