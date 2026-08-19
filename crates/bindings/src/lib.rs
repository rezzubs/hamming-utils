//! Python bindings for the various faultforge crates

mod cep;
mod common;
mod fault;
mod fault_injection;
mod mset;
mod picker;
mod secded;
mod systolic;

use pyo3::pymodule;

#[pymodule]
mod _rust {
    use super::*;

    #[pymodule]
    mod secded {
        #[pymodule_export]
        use crate::secded::PyEncoding;
        #[pymodule_export]
        use crate::secded::encode_f32;
        #[pymodule_export]
        use crate::secded::encode_u16;
    }

    #[pymodule]
    mod mset {
        #[pymodule_export]
        use crate::mset::decode_f32;
        #[pymodule_export]
        use crate::mset::decode_u16;
        #[pymodule_export]
        use crate::mset::encode_f32;
        #[pymodule_export]
        use crate::mset::encode_u16;
    }

    #[pymodule]
    mod cep {
        #[pymodule_export]
        use crate::cep::PyScheme;
        #[pymodule_export]
        use crate::cep::decode_f32;
        #[pymodule_export]
        use crate::cep::decode_u16;
        #[pymodule_export]
        use crate::cep::encode_f32;
        #[pymodule_export]
        use crate::cep::encode_u16;
    }

    #[pymodule]
    mod systolic {
        #[pymodule_export]
        use crate::systolic::AccumulatorFaultPart;
        #[pymodule_export]
        use crate::systolic::ArrayConfig;
        #[pymodule_export]
        use crate::systolic::Fault;
        #[pymodule_export]
        use crate::systolic::Index2;
        #[pymodule_export]
        use crate::systolic::LiftedFault;
        #[pymodule_export]
        use crate::systolic::Mapping;
        #[pymodule_export]
        use crate::systolic::Pass;
        #[pymodule_export]
        use crate::systolic::PeRegisterKind;
        #[pymodule_export]
        use crate::systolic::StuckAtKind;
        #[pymodule_export]
        use crate::systolic::fault_radix;
        #[pymodule_export]
        use crate::systolic::simulated_matmul;
    }

    #[pymodule_export]
    use crate::fault::PyFault;

    #[pymodule_export]
    use crate::picker::PyPicker;

    #[pymodule_export]
    use crate::fault_injection::list_of_array_fault_f32;
    #[pymodule_export]
    use crate::fault_injection::list_of_array_fault_u8;
    #[pymodule_export]
    use crate::fault_injection::list_of_array_fault_u16;
    #[pymodule_export]
    use crate::fault_injection::list_of_array_faults_f32;
    #[pymodule_export]
    use crate::fault_injection::list_of_array_faults_u8;
    #[pymodule_export]
    use crate::fault_injection::list_of_array_faults_u16;
}
