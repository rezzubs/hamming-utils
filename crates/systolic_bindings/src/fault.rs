use std::collections::HashSet;

use pyo3::{exceptions::PyValueError, prelude::*};
use systolic::fault::{
    PeRegister, PeRegisterFault, RegisterFault, RegisterFaultContext, RegisterSubset, StuckAt,
    TargetedFault,
};
use systolic::{Index2 as RustIndex2, Space};

use super::index::{ArrayConfig, Index2};

/// Which register of a processing element a fault targets.
#[pyclass(eq, hash, frozen, from_py_object, name = "PeRegisterKind")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PeRegisterKind {
    Activation,
    Weight,
    Accumulator,
}

impl From<PeRegister> for PeRegisterKind {
    fn from(value: PeRegister) -> Self {
        match value {
            PeRegister::Activation => PeRegisterKind::Activation,
            PeRegister::Weight => PeRegisterKind::Weight,
            PeRegister::Accumulator => PeRegisterKind::Accumulator,
        }
    }
}

impl From<PeRegisterKind> for PeRegister {
    fn from(value: PeRegisterKind) -> Self {
        match value {
            PeRegisterKind::Activation => PeRegister::Activation,
            PeRegisterKind::Weight => PeRegister::Weight,
            PeRegisterKind::Accumulator => PeRegister::Accumulator,
        }
    }
}

/// Whether a stuck bit is forced to zero or one.
#[pyclass(eq, hash, frozen, from_py_object, name = "StuckAtKind")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StuckAtKind {
    Zero,
    One,
}

impl From<StuckAt> for StuckAtKind {
    fn from(value: StuckAt) -> Self {
        match value {
            StuckAt::Zero => StuckAtKind::Zero,
            StuckAt::One => StuckAtKind::One,
        }
    }
}

impl From<StuckAtKind> for StuckAt {
    fn from(value: StuckAtKind) -> Self {
        match value {
            StuckAtKind::Zero => StuckAt::Zero,
            StuckAtKind::One => StuckAt::One,
        }
    }
}

fn register_fault_context(
    array: ArrayConfig,
    allowed_registers: HashSet<PeRegisterKind>,
) -> RegisterFaultContext {
    RegisterFaultContext {
        array: array.0,
        registers: RegisterSubset::new(allowed_registers.into_iter().map(PeRegister::from)),
    }
}

/// A single fault in array space. Execution-agnostic: it knows how to
/// enumerate itself (see `from_id`/`to_id`) but nothing about how it will
/// be realized (simulated vs. lifted).
#[pyclass(eq, hash, skip_from_py_object, name = "Fault")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Fault {
    /// A stuck-at fault in one register of one processing element.
    Register {
        target: Index2,
        register: PeRegisterKind,
        stuck_at: StuckAtKind,
        bit_index: u8,
    },
}

impl From<TargetedFault<PeRegisterFault>> for Fault {
    fn from(value: TargetedFault<PeRegisterFault>) -> Self {
        Fault::Register {
            target: value.target.into(),
            register: value.fault.register.into(),
            stuck_at: value.fault.fault.stuck_at.into(),
            bit_index: value.fault.fault.bit_index,
        }
    }
}

impl Fault {
    /// Reconstruct the Rust-side targeted register fault this value describes.
    /// Lifting/hook construction never needs a [`RegisterFaultContext`] - only
    /// enumeration (`to_id`/`from_id`) does.
    pub(crate) fn to_targeted(&self) -> TargetedFault<PeRegisterFault> {
        let Fault::Register {
            target,
            register,
            stuck_at,
            bit_index,
        } = *self;
        TargetedFault {
            target: RustIndex2::from(target),
            fault: PeRegisterFault {
                register: register.into(),
                fault: RegisterFault {
                    stuck_at: stuck_at.into(),
                    bit_index,
                },
            },
        }
    }
}

#[pymethods]
impl Fault {
    /// Reconstruct a fault from its id within the (possibly
    /// register-restricted) fault space over `array`.
    #[staticmethod]
    fn from_id(
        id: u64,
        array: ArrayConfig,
        allowed_registers: HashSet<PeRegisterKind>,
    ) -> PyResult<Self> {
        let context = register_fault_context(array, allowed_registers);
        let count = TargetedFault::<PeRegisterFault>::count(context);
        if id >= count {
            return Err(PyValueError::new_err(format!(
                "id {id} is out of range 0..{count}"
            )));
        }
        Ok(Fault::from(TargetedFault::<PeRegisterFault>::from_index(
            id, context,
        )))
    }

    /// This fault's id within the (possibly register-restricted) fault
    /// space over `array`.
    fn to_id(
        &self,
        array: ArrayConfig,
        allowed_registers: HashSet<PeRegisterKind>,
    ) -> PyResult<u64> {
        let targeted = self.to_targeted();
        let context = register_fault_context(array, allowed_registers);
        if !context.registers.contains(&targeted.fault.register) {
            return Err(PyValueError::new_err(
                "this fault's register is not part of `allowed_registers`",
            ));
        }
        Ok(targeted.to_index(context))
    }
}

/// The total number of distinct register faults over `array`, restricted to
/// `allowed_registers`.
#[pyfunction]
pub fn fault_radix(array: ArrayConfig, allowed_registers: HashSet<PeRegisterKind>) -> u64 {
    let context = register_fault_context(array, allowed_registers);
    TargetedFault::<PeRegisterFault>::count(context)
}
