use super::Index2;
use std::ops::{Add, Mul};

use crate::fault::PeRegister;

/// A hook invoked by the array's run loop for every register write and
/// multiply-add, letting a processing element behave differently.
///
/// Both methods pass their value through unchanged by default, so an
/// implementation can override just the one it needs - including neither, to
/// only observe values without changing anything (for example, recording what a
/// PE saw rather than injecting a fault into it).
pub trait PeHook<T> {
    /// Transform a value as it is written to a register. Called for every
    /// write in both weight loading and the run loop.
    fn on_write(&mut self, index: Index2, reg: PeRegister, v: T) -> T {
        let _ = (index, reg);
        v
    }

    /// Compute (or corrupt, or simply observe) the multiply-add of a PE. The
    /// default is the correct result; override to inject a logic fault.
    fn multiply_add(&mut self, index: Index2, activation: T, weight: T, partial_sum: T) -> T
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        let _ = index;
        activation * weight + partial_sum
    }
}

/// Zero-sized no-op hook. The default path compiles to today's code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoOp;
impl<T> PeHook<T> for NoOp {}
