use crate::{Index2, Space, mixed_radix, space};
use std::ops::{Add, Mul};

use crate::array::PeHook;

/// A gate-level fault in a PE's multiply-add unit, enumerable via netlist case index.
///
/// The radix of `Space::count` depends on the number of cases in the loaded netlist,
/// which is only known at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulatedFault {
    pub target: Index2,
    pub case: u64,
}

/// Context for enumerating [`SimulatedFault`]s.
#[derive(Clone, Copy)]
pub struct SimulatedFaultContext {
    pub array: space::ArrayConfig,
    /// Number of distinct fault cases the loaded netlist produces.
    pub sim_cases: u64,
}

impl Space for SimulatedFault {
    type Context = SimulatedFaultContext;

    fn count(context: Self::Context) -> u64 {
        Index2::count(context.array) * context.sim_cases
    }

    fn to_index(&self, context: Self::Context) -> u64 {
        mixed_radix::encode(
            [self.target.to_index(context.array), self.case],
            [Index2::count(context.array), context.sim_cases],
        )
        .expect("SimulatedFault components must be within their respective radixes")
    }

    fn from_index(index: u64, context: Self::Context) -> Self {
        let [target_index, case] =
            mixed_radix::decode(index, [Index2::count(context.array), context.sim_cases])
                .expect("index must be in 0..count");
        Self {
            target: Index2::from_index(target_index, context.array),
            case,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimulatedMulAddHook {
    pub target: Index2,
    pub case: u64,
}

impl SimulatedMulAddHook {
    pub fn from_fault(fault: SimulatedFault) -> Self {
        Self {
            target: fault.target,
            case: fault.case,
        }
    }
}

impl<T> PeHook<T> for SimulatedMulAddHook {
    fn multiply_add(&mut self, index: Index2, activation: T, weight: T, partial_sum: T) -> T
    where
        T: Add<Output = T> + Mul<Output = T>,
    {
        if index == self.target {
            todo!("simulator not yet embedded: case {}", self.case)
        } else {
            activation * weight + partial_sum
        }
    }
}
