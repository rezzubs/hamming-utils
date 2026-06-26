//! Various primitive components.

mod adder;
mod compound;
mod gate;
mod misc;
pub(crate) mod port;
mod two_terminal;

use crate::{Signal, simulation::WireId};

pub(crate) use adder::{FullAdder, HalfAdder};
pub(crate) use compound::{
    AndOrInvert21, AndOrInvert22, AndOrInvert211, AndOrInvert221, AndOrInvert222, OrAndInvert21,
    OrAndInvert22, OrAndInvert33, OrAndInvert211, OrAndInvert221, OrAndInvert222,
};
pub(crate) use gate::Gate;
pub use gate::GateKind;
pub(crate) use misc::{DLatch, Mux2};
pub(crate) use port::OutputPort;
pub(crate) use two_terminal::{Buffer, Inverter};

pub(crate) trait Component {
    /// Make this component re-evaluate its outputs.
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId);

    /// Read the value from the nth output port of this component.
    fn nth_output(&self, n: u8) -> Option<&OutputPort>;
}

/// An enum of all primitive components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GenericComponent {
    Gate2(Gate<2>),
    Gate3(Gate<3>),
    Gate4(Gate<4>),
    Buffer(Buffer),
    Inverter(Inverter),
    DLatch(DLatch),
    Mux2(Mux2),
    Input(OutputPort),
    AndOrInvert21(AndOrInvert21),
    AndOrInvert22(AndOrInvert22),
    AndOrInvert211(AndOrInvert211),
    AndOrInvert221(AndOrInvert221),
    AndOrInvert222(AndOrInvert222),
    OrAndInvert21(OrAndInvert21),
    OrAndInvert22(OrAndInvert22),
    OrAndInvert211(OrAndInvert211),
    OrAndInvert221(OrAndInvert221),
    OrAndInvert222(OrAndInvert222),
    OrAndInvert33(OrAndInvert33),
    HalfAdder(HalfAdder),
    FullAdder(FullAdder),
}

impl Component for GenericComponent {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        match self {
            GenericComponent::Gate2(gate) => gate.update(read_wire, trigger_wire_update),
            GenericComponent::Gate3(gate) => gate.update(read_wire, trigger_wire_update),
            GenericComponent::Gate4(gate) => gate.update(read_wire, trigger_wire_update),
            GenericComponent::DLatch(dlatch) => dlatch.update(read_wire, trigger_wire_update),
            GenericComponent::Mux2(mux) => mux.update(read_wire, trigger_wire_update),
            GenericComponent::Buffer(two_terminal) => {
                two_terminal.update(read_wire, trigger_wire_update)
            }
            GenericComponent::Inverter(two_terminal) => {
                two_terminal.update(read_wire, trigger_wire_update)
            }
            GenericComponent::Input(_) => {
                // Nothing to update as there are not input ports for the input component.
            }
            GenericComponent::AndOrInvert21(c) => c.update(read_wire, trigger_wire_update),
            GenericComponent::AndOrInvert22(c) => c.update(read_wire, trigger_wire_update),
            GenericComponent::AndOrInvert211(c) => c.update(read_wire, trigger_wire_update),
            GenericComponent::AndOrInvert221(c) => c.update(read_wire, trigger_wire_update),
            GenericComponent::AndOrInvert222(c) => c.update(read_wire, trigger_wire_update),
            GenericComponent::OrAndInvert21(c) => c.update(read_wire, trigger_wire_update),
            GenericComponent::OrAndInvert22(c) => c.update(read_wire, trigger_wire_update),
            GenericComponent::OrAndInvert211(c) => c.update(read_wire, trigger_wire_update),
            GenericComponent::OrAndInvert221(c) => c.update(read_wire, trigger_wire_update),
            GenericComponent::OrAndInvert222(c) => c.update(read_wire, trigger_wire_update),
            GenericComponent::OrAndInvert33(c) => c.update(read_wire, trigger_wire_update),
            GenericComponent::HalfAdder(adder) => adder.update(read_wire, trigger_wire_update),
            GenericComponent::FullAdder(adder) => adder.update(read_wire, trigger_wire_update),
        }
    }

    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match self {
            GenericComponent::Gate2(gate) => gate.nth_output(n),
            GenericComponent::Gate3(gate) => gate.nth_output(n),
            GenericComponent::Gate4(gate) => gate.nth_output(n),
            GenericComponent::DLatch(dlatch) => dlatch.nth_output(n),
            GenericComponent::Mux2(mux) => mux.nth_output(n),
            GenericComponent::Buffer(two_terminal) => two_terminal.nth_output(n),
            GenericComponent::Inverter(two_terminal) => two_terminal.nth_output(n),
            GenericComponent::Input(output_port) => match n {
                0 => Some(output_port),
                _ => None,
            },
            GenericComponent::AndOrInvert21(c) => c.nth_output(n),
            GenericComponent::AndOrInvert22(c) => c.nth_output(n),
            GenericComponent::AndOrInvert211(c) => c.nth_output(n),
            GenericComponent::AndOrInvert221(c) => c.nth_output(n),
            GenericComponent::AndOrInvert222(c) => c.nth_output(n),
            GenericComponent::OrAndInvert21(c) => c.nth_output(n),
            GenericComponent::OrAndInvert22(c) => c.nth_output(n),
            GenericComponent::OrAndInvert211(c) => c.nth_output(n),
            GenericComponent::OrAndInvert221(c) => c.nth_output(n),
            GenericComponent::OrAndInvert222(c) => c.nth_output(n),
            GenericComponent::OrAndInvert33(c) => c.nth_output(n),
            GenericComponent::HalfAdder(adder) => adder.nth_output(n),
            GenericComponent::FullAdder(adder) => adder.nth_output(n),
        }
    }
}
