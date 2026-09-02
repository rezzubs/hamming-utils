mod random;
mod register;
mod register_lift;
mod simulated;

pub use random::{RandomFault, XorMaskHook};
pub use register::{
    PeRegister, PeRegisterFault, RegisterFault, RegisterFaultContext, RegisterHook, RegisterSubset,
    StuckAt, TargetedFault,
};
pub use register_lift::{AccumulatorFaultPart, LiftedRegisterFault, LiftedRegisterFaultData};
pub use simulated::{SimulatedFault, SimulatedFaultContext, SimulatedMulAddHook};
