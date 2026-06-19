use crate::{Space, Index2, helper::u64, id, mixed_radix};
use memory::BitBuffer;

use super::hook::FaultHook;

/// A fault targeting a specific processing element (PE).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetedFault<T> {
    pub fault: T,
    pub target: Index2,
}

impl<T: Space<Context = id::ArrayConfig>> Space for TargetedFault<T> {
    type Context = id::ArrayConfig;

    fn count(context: id::ArrayConfig) -> u64 {
        Index2::count(context) * T::count(context)
    }

    fn to_index(&self, context: id::ArrayConfig) -> u64 {
        mixed_radix::encode(
            [self.target.to_index(context), self.fault.to_index(context)],
            [Index2::count(context), T::count(context)],
        )
        .expect("TargetedFault components must be within their respective radixes")
    }

    fn from_index(index: u64, context: id::ArrayConfig) -> Self {
        let [target_index, fault_index] = mixed_radix::decode(
            index,
            [Index2::count(context), T::count(context)],
        )
        .expect("index must be in 0..count");
        Self {
            target: Index2::from_index(target_index, context),
            fault: T::from_index(fault_index, context),
        }
    }
}

/// Whether a register bit is stuck at zero or one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StuckAt {
    /// The bit is stuck at zero.
    Zero = 0,
    /// The bit is stuck at one.
    One = 1,
}

impl Space for StuckAt {
    type Context = ();

    fn count(_: ()) -> u64 {
        2
    }

    fn to_index(&self, _: ()) -> u64 {
        match self {
            StuckAt::Zero => 0,
            StuckAt::One => 1,
        }
    }

    fn from_index(index: u64, _: ()) -> Self {
        match index {
            0 => StuckAt::Zero,
            1 => StuckAt::One,
            _ => panic!("index out of range for StuckAt"),
        }
    }
}

/// A stuck-at fault in a specific bit of a PE register.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RegisterFault {
    pub stuck_at: StuckAt,
    pub bit_index: u8,
}

impl Space for RegisterFault {
    type Context = id::ArrayConfig;

    fn count(context: id::ArrayConfig) -> u64 {
        StuckAt::count(()) * u64(context.dtype_bits())
    }

    fn to_index(&self, context: id::ArrayConfig) -> u64 {
        mixed_radix::encode(
            [self.stuck_at.to_index(()), u64(self.bit_index)],
            [StuckAt::count(()), u64(context.dtype_bits())],
        )
        .expect("bit_index must be less than dtype_bits")
    }

    fn from_index(index: u64, context: id::ArrayConfig) -> Self {
        let [stuck_at_index, bit_index] = mixed_radix::decode(
            index,
            [StuckAt::count(()), u64(context.dtype_bits())],
        )
        .expect("index must be in 0..count");
        Self {
            stuck_at: StuckAt::from_index(stuck_at_index, ()),
            bit_index: bit_index as u8,
        }
    }
}

/// Which register of a PE is faulty.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PeFaultRegister {
    Activation,
    Weight,
    Accumulator,
}

impl Space for PeFaultRegister {
    type Context = ();

    fn count(_: ()) -> u64 {
        3
    }

    fn to_index(&self, _: ()) -> u64 {
        match self {
            PeFaultRegister::Activation => 0,
            PeFaultRegister::Weight => 1,
            PeFaultRegister::Accumulator => 2,
        }
    }

    fn from_index(index: u64, _: ()) -> Self {
        match index {
            0 => PeFaultRegister::Activation,
            1 => PeFaultRegister::Weight,
            2 => PeFaultRegister::Accumulator,
            _ => panic!("index out of range for PeFaultRegister"),
        }
    }
}

/// A stuck-at fault targeting a specific register of a PE.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PeRegisterFault {
    pub register: PeFaultRegister,
    pub fault: RegisterFault,
}

impl Space for PeRegisterFault {
    type Context = id::ArrayConfig;

    fn count(context: id::ArrayConfig) -> u64 {
        PeFaultRegister::count(()) * RegisterFault::count(context)
    }

    fn to_index(&self, context: id::ArrayConfig) -> u64 {
        mixed_radix::encode(
            [self.register.to_index(()), self.fault.to_index(context)],
            [PeFaultRegister::count(()), RegisterFault::count(context)],
        )
        .expect("PeRegisterFault components must be within their respective radixes")
    }

    fn from_index(index: u64, context: id::ArrayConfig) -> Self {
        let [register_index, fault_index] = mixed_radix::decode(
            index,
            [PeFaultRegister::count(()), RegisterFault::count(context)],
        )
        .expect("index must be in 0..count");
        Self {
            register: PeFaultRegister::from_index(register_index, ()),
            fault: RegisterFault::from_index(fault_index, context),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterHook {
    pub target: Index2,
    pub register: PeFaultRegister,
    pub bit_index: u8,
    pub stuck_at: StuckAt,
}

impl RegisterHook {
    pub fn from_fault(fault: TargetedFault<PeRegisterFault>) -> Self {
        Self {
            target: fault.target,
            register: fault.fault.register,
            bit_index: fault.fault.fault.bit_index,
            stuck_at: fault.fault.fault.stuck_at,
        }
    }
}

impl<T: BitBuffer> FaultHook<T> for RegisterHook {
    fn on_write(&mut self, index: Index2, reg: PeFaultRegister, mut v: T) -> T {
        if index == self.target && reg == self.register {
            match self.stuck_at {
                StuckAt::Zero => v.set_0(self.bit_index as usize),
                StuckAt::One => v.set_1(self.bit_index as usize),
            }
        }
        v
    }
}
