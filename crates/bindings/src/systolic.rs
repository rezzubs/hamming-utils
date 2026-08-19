//! Python bindings for the `systolic` crate: register faults on a
//! weight-stationary systolic array, both simulated (the cycle-accurate
//! oracle) and lifted (the torch-side workhorse's Rust-computed description).

mod array;
mod fault;
mod index;
mod lift;
mod mapping;

pub use array::simulated_matmul;
pub use fault::{Fault, PeRegisterKind, StuckAtKind, fault_radix};
pub use index::{ArrayConfig, Index2};
pub use lift::{AccumulatorFaultPart, LiftedFault};
pub use mapping::{Mapping, Pass};
