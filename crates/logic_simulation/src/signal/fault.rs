//! Faulty [`Signal`]s.

use super::Signal;

/// A trait for [`Signal`] fault models.
pub trait Fault {
    /// Apply a fault on the signal..
    fn make_faulty(&self, signal: Signal) -> Signal;
}

impl Fault for () {
    /// No fault is applied, the signal is returned as-is.
    fn make_faulty(&self, signal: Signal) -> Signal {
        signal
    }
}

/// A stuck-at fault that always returns a fixed signal value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StuckAtFault {
    /// Permanent `1`.
    High,
    /// Permanent `0`.
    Low,
}

impl Fault for StuckAtFault {
    fn make_faulty(&self, signal: Signal) -> Signal {
        _ = signal;
        match self {
            StuckAtFault::High => Signal::High,
            StuckAtFault::Low => Signal::Low,
        }
    }
}

/// A fault which flips the signal value. `X` remains `X`.
pub struct FlipFault;

impl Fault for FlipFault {
    fn make_faulty(&self, signal: Signal) -> Signal {
        signal.invert()
    }
}
