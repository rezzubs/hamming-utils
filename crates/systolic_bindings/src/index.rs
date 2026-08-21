use pyo3::{exceptions::PyValueError, prelude::*};

/// A coordinate of a processing element in the array.
#[pyclass(eq, hash, frozen, from_py_object, name = "Index2")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Index2 {
    #[pyo3(get)]
    pub x: u16,
    #[pyo3(get)]
    pub y: u16,
}

#[pymethods]
impl Index2 {
    #[new]
    fn new(x: u16, y: u16) -> Self {
        Self { x, y }
    }

    fn __repr__(&self) -> String {
        format!("Index2(x={}, y={})", self.x, self.y)
    }
}

impl From<systolic::Index2> for Index2 {
    fn from(value: systolic::Index2) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<Index2> for systolic::Index2 {
    fn from(value: Index2) -> Self {
        systolic::Index2 {
            x: value.x,
            y: value.y,
        }
    }
}

/// The geometry of a systolic array and the width of its data type, as
/// needed to compute a register-fault radix.
#[pyclass(eq, hash, frozen, from_py_object, name = "ArrayConfig")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArrayConfig(pub systolic::ArrayConfig);

#[pymethods]
impl ArrayConfig {
    #[new]
    fn new(nrows: usize, ncols: usize, dtype_bits: u8) -> PyResult<Self> {
        if nrows == 0 || ncols == 0 || dtype_bits == 0 {
            return Err(PyValueError::new_err(
                "nrows, ncols, and dtype_bits must all be greater than zero",
            ));
        }
        Ok(Self(systolic::ArrayConfig::new(nrows, ncols, dtype_bits)))
    }

    #[getter]
    fn nrows(&self) -> u64 {
        self.0.nrows()
    }

    #[getter]
    fn ncols(&self) -> u64 {
        self.0.ncols()
    }

    #[getter]
    fn dtype_bits(&self) -> u8 {
        self.0.dtype_bits()
    }
}
