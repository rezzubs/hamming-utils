//! Python bindings for the various faultforge crates

mod cep;
mod common;
mod fault;
mod fault_injection;
mod mset;
mod picker;
mod secded;

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
