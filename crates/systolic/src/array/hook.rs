use super::{Index2, Pass};
use std::ops::{Add, Mul};

use crate::fault::PeRegister;

/// A hook invoked by the array's run loop for every register write and
/// multiply-add, letting a processing element behave differently.
///
/// Both `on_write` and `multiply_add` pass their value through unchanged by
/// default, so an implementation can override just the one it needs - including
/// neither, to only observe values without changing anything (for example,
/// recording what a PE saw rather than injecting a fault into it).
pub trait PeHook<T> {
    /// Called once per pass, in [`SystolicArray::matmul`], before that pass's
    /// weights/activations are loaded and run.
    ///
    /// `batch_size` is the activation matrix's column count.
    ///
    /// A no-op by default. Exists so a
    /// hook that needs to know the current pass's geometry (for example, to
    /// classify which [`Regime`] a PE is in) can pick it up
    /// without `matmul` itself knowing anything about that hook's internals.
    ///
    /// [`SystolicArray::matmul`]: crate::array::SystolicArray::matmul
    /// [`Regime`]: crate::array::Regime
    fn on_pass_start(&mut self, pass: &Pass, batch_size: usize) {
        let _ = (pass, batch_size);
    }

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
