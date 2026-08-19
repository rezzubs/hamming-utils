use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use super::fault::Fault;
use super::lift::LiftedFault;

/// How a matrix multiplication is mapped onto a systolic array, as an
/// ordered series of passes. Also the entry point for lifting a register
/// fault to matrix space (see `lift`).
#[pyclass(skip_from_py_object, name = "Mapping")]
#[derive(Debug, Clone)]
pub struct Mapping(pub systolic::Mapping);

#[pymethods]
impl Mapping {
    /// Automatically map `weights` (`out_features, in_features`) onto an
    /// `array_nrows x array_ncols` array, splitting into multiple passes if
    /// the weights don't fit in one.
    #[staticmethod]
    fn auto_for(weights: PyReadonlyArray2<f32>, array_nrows: usize, array_ncols: usize) -> Self {
        let weights = weights.as_array();
        Self(systolic::Mapping::auto_for(
            &weights,
            array_nrows,
            array_ncols,
        ))
    }

    /// Lift a register fault targeting this mapping's array to matrix
    /// space. The result describes an equivalent operation on the
    /// weight/activation/output matrices rather than the array's physical
    /// registers.
    fn lift(&self, fault: &Fault) -> LiftedFault {
        let targeted = fault.to_targeted();
        self.0.lift_register_fault(&targeted).data.into()
    }
}
