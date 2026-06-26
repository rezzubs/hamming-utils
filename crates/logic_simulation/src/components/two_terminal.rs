use std::marker::PhantomData;

use crate::{
    Signal,
    components::{
        Component,
        port::{InputPort, OutputPort},
    },
    simulation::WireId,
};

pub(crate) trait TwoTerminalFn {
    fn apply(signal: Signal) -> Signal;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Invert;

impl TwoTerminalFn for Invert {
    fn apply(signal: Signal) -> Signal {
        signal.invert()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Identity;

impl TwoTerminalFn for Identity {
    fn apply(signal: Signal) -> Signal {
        signal
    }
}

/// A simple buffer that relays the input to its output,
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TwoTerminal<F> {
    input: InputPort,
    output: OutputPort,
    _fn: PhantomData<F>,
}

impl<F> TwoTerminal<F> {
    pub fn new(input: InputPort, output: OutputPort) -> Self {
        Self {
            input,
            output,
            _fn: PhantomData,
        }
    }
}

impl<F: TwoTerminalFn> Component for TwoTerminal<F> {
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: FnOnce(WireId) -> Signal,
        T: FnMut(WireId),
    {
        let value = F::apply(self.input.read(read_wire));
        self.output.write(value, trigger_wire_update);
    }

    /// Read the nth output port.
    ///
    /// This component only has a single port, 0 is the only valid value for
    /// `n`.
    fn nth_output(&self, n: u8) -> Option<&OutputPort> {
        match n {
            0 => Some(&self.output),
            _ => None,
        }
    }
}

/// A two-terminal component that inverts the input signal.
pub(crate) type Inverter = TwoTerminal<Invert>;

/// A buffer that relays the input to its output.
pub(crate) type Buffer = TwoTerminal<Identity>;
