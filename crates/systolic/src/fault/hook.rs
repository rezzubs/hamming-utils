use crate::Index2;
use std::ops::{Add, Mul};

use super::register::PeFaultRegister;

/// A hook invoked by the array's run loop, allowing a single PE to behave differently.
///
/// Both methods have no-op defaults so implementations only override what they need.
pub trait FaultHook<T> {
    /// Transform a value as it is written to a register. Called for every write
    /// in both weight loading and the run loop.
    fn on_write(&mut self, index: Index2, reg: PeFaultRegister, v: T) -> T {
        let _ = (index, reg);
        v
    }

    /// Compute (or corrupt) the multiply-add of a PE. The default is the correct
    /// result; override to inject a logic fault.
    fn multiply_add(&mut self, index: Index2, activation: T, weight: T, partial_sum: T) -> T
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        let _ = index;
        activation * weight + partial_sum
    }
}

/// Zero-sized no-op hook. The fault-free path compiles to today's code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoFault;
impl<T> FaultHook<T> for NoFault {}
