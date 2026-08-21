use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use systolic::SystolicArray;
use systolic::fault::RegisterHook;

use super::fault::Fault;
use super::mapping::Mapping;

fn new_array(nrows: usize, ncols: usize) -> PyResult<SystolicArray<f32>> {
    SystolicArray::<f32>::new(nrows, ncols)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn check_mapping<H>(array: &SystolicArray<f32, H>, mapping: &systolic::Mapping) -> PyResult<()> {
    if array.supports_mapping(mapping) {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "mapping requires an array larger than (array_nrows, array_ncols)",
        ))
    }
}

/// Run one matmul through a literal, cycle-accurate systolic array
/// simulation, optionally with one register fault applied.
///
/// This is the oracle: correct by construction (it's the same run loop the
/// physical array would execute), but pays for that by simulating cycle by
/// cycle, so it's slow relative to the lifted torch path. f32 only.
#[pyfunction]
pub fn simulated_matmul<'py>(
    py: Python<'py>,
    mapping: &Mapping,
    weights: PyReadonlyArray2<f32>,
    activations: PyReadonlyArray2<f32>,
    array_nrows: usize,
    array_ncols: usize,
    fault: Option<&Fault>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let weights = weights.as_array();
    let activations = activations.as_array();

    // `SystolicArray<T, H>` is generic over the hook type at compile time,
    // so the two branches build differently-typed arrays; both produce an
    // `Array2<f32>` from `.matmul`, which is all that needs to unify.
    let result = match fault {
        None => {
            let mut array = new_array(array_nrows, array_ncols)?;
            check_mapping(&array, &mapping.0)?;
            array.matmul(&mapping.0, &weights, &activations)
        }
        Some(fault) => {
            let hook = RegisterHook::from_fault(fault.to_targeted());
            let mut array = new_array(array_nrows, array_ncols)?.with_hook(hook);
            check_mapping(&array, &mapping.0)?;
            array.matmul(&mapping.0, &weights, &activations)
        }
    };

    Ok(result.into_pyarray(py))
}
