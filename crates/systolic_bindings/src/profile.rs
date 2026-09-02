//! Bindings for `systolic::profile`: run a matmul while recording a per-PE,
//! per-regime sample of the MAC inputs the array sees, and pad the result
//! into the dense, on-disk shape `systolic-roadmap.md` settles on.
//!
//! Ragged-to-dense padding happens here, not in `systolic::profile` (which
//! stays ragged by design) and not in Python.

use numpy::ndarray::{Array2, Array3, Array4};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use systolic::SystolicArray;
use systolic::profile::{LogicInput, RecordingHook};

use super::array::{check_mapping, new_array};
use super::mapping::Mapping;

/// Accumulates a profiling artifact across many matmul calls - one per
/// (layer, batch) pair over a model's forward pass - sharing one physical
/// array and one set of per-PE reservoirs across all of them.
///
/// Profiling is a per-(model, dataset, array) precompute: every layer of a
/// model reuses the same physical array, just with a different `Mapping`
/// and weight shape, so their observations belong in one pooled reservoir
/// per PE rather than one artifact per layer.
#[pyclass]
pub struct Profiler {
    /// The array being profiled, with the recording hook installed.
    array: SystolicArray<f32, RecordingHook<StdRng>>,
    /// Per-PE, per-regime reservoir capacity, kept alongside the array so
    /// `artifact()` can pad ragged samples out to this width without
    /// needing a getter on `RecordingHook` itself.
    capacity: usize,
}

#[pymethods]
impl Profiler {
    /// Creates a profiler for an `array_nrows x array_ncols` array, with
    /// every PE's per-regime reservoir bounded to `capacity` samples and a
    /// single RNG stream (seeded from `seed`) shared across every PE and
    /// regime.
    #[new]
    fn new(array_nrows: usize, array_ncols: usize, capacity: usize, seed: u64) -> PyResult<Self> {
        let hook = RecordingHook::new(
            array_nrows,
            array_ncols,
            capacity,
            StdRng::seed_from_u64(seed),
        );
        Ok(Self {
            array: new_array(array_nrows, array_ncols)?.with_hook(hook),
            capacity,
        })
    }

    /// Runs one matmul through the same cycle-accurate simulation as
    /// `simulated_matmul`, accumulating into the shared reservoirs. Returns
    /// the plain result so a caller can feed real activations through the
    /// rest of the model (the next layer needs this layer's actual,
    /// post-activation-function output, not independently generated data).
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        mapping: &Mapping,
        weights: PyReadonlyArray2<f32>,
        activations: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let weights = weights.as_array();
        let activations = activations.as_array();
        check_mapping(&self.array, &mapping.0)?;
        let result = self.array.matmul(&mapping.0, &weights, &activations);
        Ok(result.into_pyarray(py))
    }

    /// Snapshots everything accumulated so far into a dense, serializable
    /// artifact. Non-destructive - callable more than once, e.g. to
    /// checkpoint mid-run.
    fn artifact(&self, py: Python<'_>) -> ProfilingArtifact {
        let artifact = self.array.hook().to_artifact();
        ProfilingArtifact {
            first_triples: triples_to_dense(&artifact.first_triples, self.capacity)
                .into_pyarray(py)
                .unbind(),
            first_fill: artifact.first_fill.into_pyarray(py).unbind(),
            active_triples: triples_to_dense(&artifact.active_triples, self.capacity)
                .into_pyarray(py)
                .unbind(),
            active_fill: artifact.active_fill.into_pyarray(py).unbind(),
            drain_partial_sums: partial_sums_to_dense(&artifact.drain_partial_sums, self.capacity)
                .into_pyarray(py)
                .unbind(),
            drain_fill: artifact.drain_fill.into_pyarray(py).unbind(),
        }
    }
}

/// The dense, on-disk-ready result of one or more profiled matmuls, produced by
/// [`Profiler::artifact`].
///
/// See `systolic::profile:: ProfilingArtifact` for the ragged, in-memory
/// source this is padded from (not `use`-imported here, to avoid colliding
/// with this identically-named pyclass - matches how `mapping.rs` handles
/// `systolic::Mapping`).
///
/// `*_fill` entries are `min(capacity, observations)` per PE; anything at
/// or past `fill` along the sample axis is `0.0` padding, not a real
/// observation. Zero-regime PEs have no array here at all: nothing is
/// stored for them, since their inputs are structurally always zero.
#[pyclass(skip_from_py_object, name = "ProfilingArtifact")]
#[derive(Debug)]
pub struct ProfilingArtifact {
    /// `(array_nrows, array_ncols, capacity, 3)`: per-PE FIRST-regime
    /// `(activation, weight, partial_sum)` samples.
    #[pyo3(get)]
    pub first_triples: Py<PyArray4<f32>>,
    /// `(array_nrows, array_ncols)`: real sample count per PE in
    /// `first_triples`.
    #[pyo3(get)]
    pub first_fill: Py<PyArray2<usize>>,
    /// Same shape/meaning as `first_triples`, for the ACTIVE regime.
    #[pyo3(get)]
    pub active_triples: Py<PyArray4<f32>>,
    /// Same shape/meaning as `first_fill`, for `active_triples`.
    #[pyo3(get)]
    pub active_fill: Py<PyArray2<usize>>,
    /// `(array_nrows, array_ncols, capacity)`: per-PE DRAIN-regime incoming
    /// partial-sum samples. Activation/weight are always zero in DRAIN, so
    /// only the partial sum is kept.
    #[pyo3(get)]
    pub drain_partial_sums: Py<PyArray3<f32>>,
    /// `(array_nrows, array_ncols)`: real sample count per PE in
    /// `drain_partial_sums`.
    #[pyo3(get)]
    pub drain_fill: Py<PyArray2<usize>>,
}

/// Pads each PE's ragged `LogicInput` sample (length `<= capacity`) to
/// exactly `capacity`, trailing axes `(capacity, 3)` in `activation,
/// weight, partial_sum` order. Padding past a PE's fill count is `0.0` -
/// safe, since the matching `*_fill` array says exactly where real data
/// ends.
fn triples_to_dense(grid: &Array2<Vec<LogicInput>>, capacity: usize) -> Array4<f32> {
    let (nrows, ncols) = grid.dim();
    Array4::from_shape_fn((nrows, ncols, capacity, 3), |(y, x, k, c)| {
        grid[[y, x]].get(k).map_or(0.0, |triple| match c {
            0 => triple.activation,
            1 => triple.weight,
            2 => triple.partial_sum,
            _ => unreachable!("axis of 3"),
        })
    })
}

/// Same padding as [`triples_to_dense`], for the single-valued DRAIN
/// partial sums.
fn partial_sums_to_dense(grid: &Array2<Vec<f32>>, capacity: usize) -> Array3<f32> {
    let (nrows, ncols) = grid.dim();
    Array3::from_shape_fn((nrows, ncols, capacity), |(y, x, k)| {
        grid[[y, x]].get(k).copied().unwrap_or(0.0)
    })
}
