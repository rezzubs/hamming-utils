mod hook;
mod random;
mod register;
mod simulated;

pub use hook::{FaultHook, NoFault};
pub use random::{RandomFault, XorMaskHook};
pub use register::{
    PeFaultRegister, PeRegisterFault, RegisterFault, RegisterHook, StuckAt, TargetedFault,
};
pub use simulated::{SimulatedFault, SimulatedFaultContext, SimulatedMulAddHook};
