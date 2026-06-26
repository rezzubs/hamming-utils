//! This is a crate for running gate level digital logic simulations with faults.
//!
//! See [`Simulation`] for details.
#![warn(missing_docs)]

mod components;
pub mod netlist;
mod signal;
mod simulation;

pub use signal::{Signal, fault};
pub use simulation::{
    FaultOutOfBoundsError, InputBusLocateError, OutputBusLocateError, Simulation, WireReadError,
    builder, bus,
};
