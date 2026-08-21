//! Python bindings for the `systolic` crate: register faults on a
//! weight-stationary systolic array, both simulated (the cycle-accurate
//! oracle) and lifted (the torch-side workhorse's Rust-computed description).
//!
//! Compiled as its own extension module (`systolic._rust`) rather than
//! living inside `faultforge._rust`, so that installing `faultforge` never
//! pulls in the `systolic` crate or its compiled bindings - only the
//! `systolic` experiment package (`experiments/systolic`) depends on this.

mod array;
mod fault;
mod index;
mod lift;
mod mapping;

use pyo3::pymodule;

#[pymodule]
mod _rust {
    #[pymodule_export]
    use crate::array::simulated_matmul;
    #[pymodule_export]
    use crate::fault::Fault;
    #[pymodule_export]
    use crate::fault::PeRegisterKind;
    #[pymodule_export]
    use crate::fault::StuckAtKind;
    #[pymodule_export]
    use crate::fault::fault_radix;
    #[pymodule_export]
    use crate::index::ArrayConfig;
    #[pymodule_export]
    use crate::index::Index2;
    #[pymodule_export]
    use crate::lift::AccumulatorFaultPart;
    #[pymodule_export]
    use crate::lift::LiftedFault;
    #[pymodule_export]
    use crate::mapping::Mapping;
    #[pymodule_export]
    use crate::mapping::Pass;
}
