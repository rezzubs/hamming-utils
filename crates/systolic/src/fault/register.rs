use crate::{Index2, Space, helper::u64, mixed_radix, space, space::AsArrayConfig};
use memory::BitBuffer;

use crate::array::PeHook;

/// A fault targeting a specific processing element (PE).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetedFault<T> {
    pub fault: T,
    pub target: Index2,
}

impl<T: Space> Space for TargetedFault<T>
where
    T::Context: AsArrayConfig,
{
    type Context = T::Context;

    fn count(context: T::Context) -> u64 {
        Index2::count(context.array_config()) * T::count(context)
    }

    fn to_index(&self, context: T::Context) -> u64 {
        mixed_radix::encode(
            [
                self.target.to_index(context.array_config()),
                self.fault.to_index(context),
            ],
            [Index2::count(context.array_config()), T::count(context)],
        )
        .expect("TargetedFault components must be within their respective radixes")
    }

    fn from_index(index: u64, context: T::Context) -> Self {
        let [target_index, fault_index] = mixed_radix::decode(
            index,
            [Index2::count(context.array_config()), T::count(context)],
        )
        .expect("index must be in 0..count");
        Self {
            target: Index2::from_index(target_index, context.array_config()),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisterFault {
    pub stuck_at: StuckAt,
    pub bit_index: u8,
}

impl RegisterFault {
    /// Apply this fault to a value by corrupting the target bit.
    pub fn apply<T: memory::BitBuffer>(&self, mut value: T) -> T {
        match self.stuck_at {
            StuckAt::Zero => value.set_0(self.bit_index.into()),
            StuckAt::One => value.set_1(self.bit_index.into()),
        }
        value
    }
}

impl Space for RegisterFault {
    type Context = space::ArrayConfig;

    fn count(context: space::ArrayConfig) -> u64 {
        StuckAt::count(()) * u64(context.dtype_bits())
    }

    fn to_index(&self, context: space::ArrayConfig) -> u64 {
        mixed_radix::encode(
            [self.stuck_at.to_index(()), u64(self.bit_index)],
            [StuckAt::count(()), u64(context.dtype_bits())],
        )
        .expect("bit_index must be less than dtype_bits")
    }

    fn from_index(index: u64, context: space::ArrayConfig) -> Self {
        let [stuck_at_index, bit_index] =
            mixed_radix::decode(index, [StuckAt::count(()), u64(context.dtype_bits())])
                .expect("index must be in 0..count");
        Self {
            stuck_at: StuckAt::from_index(stuck_at_index, ()),
            bit_index: bit_index as u8,
        }
    }
}

/// Which register of a PE is faulty.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PeRegister {
    Activation,
    Weight,
    Accumulator,
}

impl Space for PeRegister {
    type Context = ();

    fn count(_: ()) -> u64 {
        3
    }

    fn to_index(&self, _: ()) -> u64 {
        match self {
            PeRegister::Activation => 0,
            PeRegister::Weight => 1,
            PeRegister::Accumulator => 2,
        }
    }

    fn from_index(index: u64, _: ()) -> Self {
        match index {
            0 => PeRegister::Activation,
            1 => PeRegister::Weight,
            2 => PeRegister::Accumulator,
            _ => panic!("index out of range for PeRegister"),
        }
    }
}

/// A restriction of which [`PeRegister`] variants are eligible for a
/// fault. Shrinks [`PeRegisterFault`]'s radix to just the registers of
/// interest via a dense re-index (not reject sampling), so the `Picker`'s
/// without-replacement exhaustion stays correct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisterSubset {
    activation: bool,
    weight: bool,
    accumulator: bool,
}

impl RegisterSubset {
    /// All three registers are eligible.
    pub const ALL: Self = Self {
        activation: true,
        weight: true,
        accumulator: true,
    };

    /// Create a new `RegisterSubset` from a sequence of register variants.
    pub fn new(registers: impl IntoIterator<Item = PeRegister>) -> Self {
        let mut subset = Self {
            activation: false,
            weight: false,
            accumulator: false,
        };
        for register in registers {
            match register {
                PeRegister::Activation => subset.activation = true,
                PeRegister::Weight => subset.weight = true,
                PeRegister::Accumulator => subset.accumulator = true,
            }
        }
        subset
    }

    /// Whether `register` is eligible under this subset.
    ///
    /// Callers that build a [`PeRegisterFault`]/[`crate::fault::TargetedFault`]
    /// from data not already known to respect a [`RegisterFaultContext`] (e.g.
    /// a value crossing an FFI boundary) should check this before indexing,
    /// since `index_of` panics on a non-member register.
    pub fn contains(&self, register: &PeRegister) -> bool {
        match register {
            PeRegister::Activation => self.activation,
            PeRegister::Weight => self.weight,
            PeRegister::Accumulator => self.accumulator,
        }
    }

    /// The eligible registers, in a fixed canonical order. This order is what
    /// makes the dense re-index deterministic and round-trippable.
    fn ordered(&self) -> impl Iterator<Item = PeRegister> + '_ {
        [
            PeRegister::Activation,
            PeRegister::Weight,
            PeRegister::Accumulator,
        ]
        .into_iter()
        .filter(move |register| self.contains(register))
    }

    fn count(&self) -> u64 {
        self.ordered().count() as u64
    }

    /// Get the index of `register` within [`Self::ordered`].
    fn index_of(&self, register: &PeRegister) -> u64 {
        self.ordered()
            .position(|candidate| &candidate == register)
            .expect("register must be a member of the subset") as u64
    }

    /// Get a corresponding register from the order defined by [`Self::ordered`].
    fn register_at(&self, index: u64) -> PeRegister {
        self.ordered()
            .nth(index as usize)
            .expect("index must be within the subset's count")
    }
}

/// Context for [`PeRegisterFault`]'s [`Space`] impl: the array geometry plus
/// which registers are eligible for a fault.
///
/// Kept separate from [`space::ArrayConfig`] itself so plain array indexing
/// (used by fault kinds with no register concept, e.g. logic faults) stays
/// register-agnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegisterFaultContext {
    pub array: space::ArrayConfig,
    pub registers: RegisterSubset,
}

impl space::AsArrayConfig for RegisterFaultContext {
    fn array_config(&self) -> space::ArrayConfig {
        self.array
    }
}

/// A stuck-at fault targeting a specific register of a PE.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PeRegisterFault {
    pub register: PeRegister,
    pub fault: RegisterFault,
}

impl Space for PeRegisterFault {
    type Context = RegisterFaultContext;

    fn count(context: RegisterFaultContext) -> u64 {
        context.registers.count() * RegisterFault::count(context.array)
    }

    fn to_index(&self, context: RegisterFaultContext) -> u64 {
        mixed_radix::encode(
            [
                context.registers.index_of(&self.register),
                self.fault.to_index(context.array),
            ],
            [
                context.registers.count(),
                RegisterFault::count(context.array),
            ],
        )
        .expect("PeRegisterFault components must be within their respective radixes")
    }

    fn from_index(index: u64, context: RegisterFaultContext) -> Self {
        let [register_index, fault_index] = mixed_radix::decode(
            index,
            [
                context.registers.count(),
                RegisterFault::count(context.array),
            ],
        )
        .expect("index must be in 0..count");
        Self {
            register: context.registers.register_at(register_index),
            fault: RegisterFault::from_index(fault_index, context.array),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterHook {
    pub target: Index2,
    pub register: PeRegister,
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

impl<T: BitBuffer> PeHook<T> for RegisterHook {
    fn on_write(&mut self, index: Index2, reg: PeRegister, mut v: T) -> T {
        if index == self.target && reg == self.register {
            match self.stuck_at {
                StuckAt::Zero => v.set_0(self.bit_index as usize),
                StuckAt::One => v.set_1(self.bit_index as usize),
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PeRegister, PeRegisterFault, RegisterFault, RegisterFaultContext, RegisterHook,
        RegisterSubset, StuckAt, TargetedFault,
    };
    use crate::Index2;
    use crate::array::PeHook;
    use crate::space::{ArrayConfig, Space};

    fn config(nrows: usize, ncols: usize, dtype_bits: u8) -> ArrayConfig {
        ArrayConfig::new(nrows, ncols, dtype_bits)
    }

    fn register_context(
        nrows: usize,
        ncols: usize,
        dtype_bits: u8,
        registers: RegisterSubset,
    ) -> RegisterFaultContext {
        RegisterFaultContext {
            array: config(nrows, ncols, dtype_bits),
            registers,
        }
    }

    #[test]
    fn stuck_at_round_trip() {
        for variant in [StuckAt::Zero, StuckAt::One] {
            assert_eq!(StuckAt::from_index(variant.to_index(()), ()), variant);
        }
        for index in 0..StuckAt::count(()) {
            assert_eq!(StuckAt::from_index(index, ()).to_index(()), index);
        }
    }

    #[test]
    fn pe_fault_register_round_trip() {
        for variant in [
            PeRegister::Activation,
            PeRegister::Weight,
            PeRegister::Accumulator,
        ] {
            assert_eq!(PeRegister::from_index(variant.to_index(()), ()), variant);
        }
        for index in 0..PeRegister::count(()) {
            assert_eq!(PeRegister::from_index(index, ()).to_index(()), index);
        }
    }

    #[test]
    fn register_fault_round_trip() {
        let context = config(1, 1, 8);
        for index in 0..RegisterFault::count(context) {
            let fault = RegisterFault::from_index(index, context);
            assert_eq!(fault.to_index(context), index);
        }
    }

    #[test]
    fn pe_register_fault_round_trip() {
        let context = register_context(1, 1, 8, RegisterSubset::ALL);
        for index in 0..PeRegisterFault::count(context) {
            let fault = PeRegisterFault::from_index(index, context);
            assert_eq!(fault.to_index(context), index);
        }
    }

    #[test]
    fn pe_register_fault_round_trip_restricted_subset() {
        let registers = RegisterSubset::new([PeRegister::Weight, PeRegister::Accumulator]);
        let context = register_context(1, 1, 8, registers);
        assert_eq!(
            PeRegisterFault::count(context),
            2 * RegisterFault::count(context.array)
        );
        for index in 0..PeRegisterFault::count(context) {
            let fault = PeRegisterFault::from_index(index, context);
            assert_ne!(fault.register, PeRegister::Activation);
            assert_eq!(fault.to_index(context), index);
        }
    }

    #[test]
    fn targeted_fault_round_trip() {
        let context = register_context(2, 2, 4, RegisterSubset::ALL);
        for index in 0..TargetedFault::<PeRegisterFault>::count(context) {
            let fault = TargetedFault::<PeRegisterFault>::from_index(index, context);
            assert_eq!(fault.to_index(context), index);
        }
    }

    #[test]
    fn targeted_fault_round_trip_restricted_subset() {
        let registers = RegisterSubset::new([PeRegister::Weight]);
        let context = register_context(2, 2, 4, registers);
        for index in 0..TargetedFault::<PeRegisterFault>::count(context) {
            let fault = TargetedFault::<PeRegisterFault>::from_index(index, context);
            assert_eq!(fault.fault.register, PeRegister::Weight);
            assert_eq!(fault.to_index(context), index);
        }
    }

    fn make_hook(
        x: u16,
        y: u16,
        register: PeRegister,
        bit_index: u8,
        stuck_at: StuckAt,
    ) -> RegisterHook {
        RegisterHook {
            target: Index2 { x, y },
            register,
            bit_index,
            stuck_at,
        }
    }

    #[test]
    fn on_write_stuck_at_zero_clears_bit() {
        let mut hook = make_hook(0, 0, PeRegister::Weight, 0, StuckAt::Zero);
        let result: u8 = hook.on_write(Index2 { x: 0, y: 0 }, PeRegister::Weight, 0xFF);
        assert_eq!(result, 0xFE);
    }

    #[test]
    fn on_write_stuck_at_one_sets_bit() {
        let mut hook = make_hook(0, 0, PeRegister::Weight, 0, StuckAt::One);
        let result: u8 = hook.on_write(Index2 { x: 0, y: 0 }, PeRegister::Weight, 0x00);
        assert_eq!(result, 0x01);
    }

    #[test]
    fn on_write_non_matching_pe_passes_through() {
        let mut hook = make_hook(0, 0, PeRegister::Weight, 0, StuckAt::Zero);
        let result: u8 = hook.on_write(Index2 { x: 1, y: 0 }, PeRegister::Weight, 0xFF);
        assert_eq!(result, 0xFF);
    }

    #[test]
    fn on_write_non_matching_register_passes_through() {
        let mut hook = make_hook(0, 0, PeRegister::Weight, 0, StuckAt::Zero);
        let result: u8 = hook.on_write(Index2 { x: 0, y: 0 }, PeRegister::Activation, 0xFF);
        assert_eq!(result, 0xFF);
    }
}
