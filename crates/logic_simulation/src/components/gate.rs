use super::port::OutputPort;
use crate::{
    Signal,
    components::{Component, port::InputPort},
    simulation::WireId,
};

/// Which logical operation a `Gate` performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateKind {
    /// A gate that performs a logical AND operation.
    And,
    /// A gate that performs a logical OR operation.
    Or,
    /// A gate that performs a logical XOR operation.
    Xor,
    /// A gate that performs a logical NAND operation.
    Nand,
    /// A gate that performs a logical NOR operation.
    Nor,
    /// A gate that performs a logical XNOR operation.
    Xnor,
}

impl GateKind {
    fn reduce(self, accumulator: Signal, input: Signal) -> Signal {
        match self {
            GateKind::And | GateKind::Nand => accumulator.and(input),
            GateKind::Or | GateKind::Nor => accumulator.or(input),
            GateKind::Xor | GateKind::Xnor => accumulator.xor(input),
        }
    }

    fn finalize(self, value: Signal) -> Signal {
        match self {
            GateKind::And | GateKind::Or | GateKind::Xor => value,
            GateKind::Nand | GateKind::Nor | GateKind::Xnor => value.invert(),
        }
    }
}

/// A logic gate that reduces N inputs to a single output via [`GateKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Gate<const N: usize> {
    pub kind: GateKind,
    pub inputs: [InputPort; N],
    pub output: OutputPort,
}

impl<const N: usize> Component for Gate<N> {
    /// Updates the output port based on the inputs.
    ///
    /// # Panics
    ///
    /// Panics if `N` is less than 2.
    fn update<R, T>(&mut self, read_wire: R, trigger_wire_update: T)
    where
        R: Fn(WireId) -> Signal,
        T: FnMut(WireId),
    {
        if N < 2 {
            panic!("Gate requires at least 2 inputs");
        }

        let kind = self.kind;
        let value = self
            .inputs
            .iter()
            .map(|input_port| input_port.read(&read_wire))
            .reduce(|accumulator, input| kind.reduce(accumulator, input))
            .expect("The length is checked above");

        let value = kind.finalize(value);

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
